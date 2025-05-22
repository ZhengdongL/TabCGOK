##合并tabr_classGroup_cat和tabr_classGroup_num
# The model.
# >>>
if __name__ == '__main__':
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ['PROJECT_DIR'] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import delu
import faiss
import faiss.contrib.torch_utils  # noqa  << this line makes faiss work with PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
from loguru import logger
from torch import Tensor
from tqdm import tqdm

import lib.ordinal_compute as ordinal_compute

import lib
from lib import KWArgs
from lib.splitClassGroup import split_class_group


@dataclass(frozen=True)
class Config:
    seed: int
    data: Union[lib.Dataset[np.ndarray], KWArgs]  # lib.data.build_dataset
    model: KWArgs  # Model
    context_size: int
    optimizer: KWArgs  # lib.deep.make_optimizer
    batch_size: int
    patience: Optional[int]
    n_epochs: Union[int, float]


def custom_function(x):
    return x.squeeze(-2)


class CustomLambda(nn.Module):
    def __init__(self, custom_fn):
        super(CustomLambda, self).__init__()
        self.custom_fn = custom_fn

    def forward(self, x):
        return self.custom_fn(x)


class Model(nn.Module):
    def __init__(
            self,
            *,
            #
            n_num_features: int,
            n_bin_features: int,
            cat_cardinalities: list[int],
            n_classes: Optional[int],
            #
            num_embeddings: Optional[dict],  # lib.deep.ModuleSpec
            d_main: int,
            d_multiplier: float,
            encoder_n_blocks: int,
            predictor_n_blocks: int,
            mixer_normalization: Union[bool, Literal['auto']],
            context_dropout: float,
            dropout0: float,
            dropout1: Union[float, Literal['dropout0']],
            normalization: str,
            activation: str,
            #
            # The following options should be used only when truly needed.
            memory_efficient: bool = False,
            candidate_encoding_batch_size: Optional[int] = None,
    ) -> None:
        if not memory_efficient:
            assert candidate_encoding_batch_size is None
        if mixer_normalization == 'auto':
            mixer_normalization = encoder_n_blocks > 0
        if encoder_n_blocks == 0:
            assert not mixer_normalization
        super().__init__()
        if dropout1 == 'dropout0':
            dropout1 = dropout0

        self.one_hot_encoder = (
            lib.OneHotEncoder(cat_cardinalities) if cat_cardinalities else None
        )
        self.num_embeddings = (
            None
            if num_embeddings is None
            else lib.make_module(num_embeddings, n_features=n_num_features)
        )

        # >>> E encoder
        d_in = (
                n_num_features
                * (1 if num_embeddings is None else num_embeddings['d_embedding'])
                + n_bin_features
                + sum(cat_cardinalities)
        )
        d_block = int(d_main * d_multiplier)
        Normalization = getattr(nn, normalization)  # 从 nn 模块中获取名为 normalization 的属性或方法
        Activation = getattr(nn, activation)

        def make_block(prenorm: bool) -> nn.Sequential:
            return nn.Sequential(
                *([Normalization(d_main)] if prenorm else []),  ### * 用于解包为单独参数
                nn.Linear(d_main, d_block),
                # *([Normalization(d_main * 2)] if prenorm else []),  # lzd
                # nn.Linear(d_main * 2, d_block),  # lzd
                Activation(),
                nn.Dropout(dropout0),
                nn.Linear(d_block, d_main),
                # nn.Linear(d_block, d_main * 2),  # lzd-cat
                nn.Dropout(dropout1),
            )

        self.linear = nn.Linear(d_in, d_main)
        self.blocks0 = nn.ModuleList(
            [make_block(i > 0) for i in range(encoder_n_blocks)]
        )

        # >>> R retrieval
        self.normalization = Normalization(d_main) if mixer_normalization else None

        self.label_encoder = (
            nn.Linear(1, d_main)
            if n_classes is None
            else nn.Sequential(
                nn.Embedding(n_classes, d_main), delu.nn.Lambda(lambda x: x.squeeze(-2))  ##pip install delu==0.0.15
                # nn.Embedding(n_classes, d_main), CustomLambda(custom_function)
            )
        )
        self.label_encoder_ = (nn.Linear(1, d_main)) ##在class_group中虽然n_classes不为none,输入的不是整数标签而是浮点值，所以不能用Embedding而用Linear
        self.K = nn.Linear(d_main, d_main)
        self.T = nn.Sequential(
            nn.Linear(d_main, d_block),
            Activation(),
            nn.Dropout(dropout0),
            nn.Linear(d_block, d_main, bias=False),
        )
        self.dropout = nn.Dropout(context_dropout)

        # >>> P  predictor
        self.blocks1 = nn.ModuleList(
            [make_block(True) for _ in range(predictor_n_blocks)]
        )
        self.head = nn.Sequential(
            Normalization(d_main),
            # Normalization(d_main * 2),  # lzd-cat
            Activation(),
            nn.Linear(d_main, lib.get_d_out(n_classes)),
            # nn.Linear(d_main * 2, lib.get_d_out(n_classes)),  # lzd-cat
        )

        # >>>
        self.search_index = None
        self.memory_efficient = memory_efficient
        self.candidate_encoding_batch_size = candidate_encoding_batch_size
        self.reset_parameters()  # 重置对象的参数或状态

    def reset_parameters(self):
        if isinstance(self.label_encoder, nn.Linear):
            bound = 1 / math.sqrt(2.0)  # 1除以2的平方根
            nn.init.uniform_(self.label_encoder.weight, -bound, bound)  # type: ignore[code]  # noqa: E501#均匀分布的初始化，-bound 和 bound 是两个边界值
            nn.init.uniform_(self.label_encoder.bias, -bound, bound)  # type: ignore[code]  # noqa: E501
        else:
            assert isinstance(self.label_encoder[0], nn.Embedding)
            nn.init.uniform_(self.label_encoder[0].weight, -1.0, 1.0)  # type: ignore[code]  # noqa: E501

    def _encode(self, x_: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        x_num = x_.get('num')
        x_bin = x_.get('bin')
        x_cat = x_.get('cat')
        del x_

        x = []
        if x_num is None:
            assert self.num_embeddings is None
        else:
            x.append(
                x_num
                if self.num_embeddings is None
                else self.num_embeddings(x_num).flatten(1)
            )
        if x_bin is not None:
            x.append(x_bin)
        if x_cat is None:
            assert self.one_hot_encoder is None
        else:
            assert self.one_hot_encoder is not None
            x.append(self.one_hot_encoder(x_cat))
        assert x
        x = torch.cat(x, dim=1)

        x = self.linear(x)
        for block in self.blocks0:
            x = x + block(x)
        k = self.K(x if self.normalization is None else self.normalization(x))
        return x, k

    def forward(
            self,
            *,
            x_: dict[str, Tensor],
            y: Optional[Tensor],
            rank, #lzd
            candidate_x_: dict[str, Tensor],
            candidate_y: Tensor,
            candidate_rank, #lzd
            context_size: int,
            data_classGroup,
            is_train: bool,
            task_type,
    ) -> Tensor:
        # >>>
        if data_classGroup[4]['cat'] is None:
            candidate_classGroup = {'num': data_classGroup[0]}
            candidate_y_classgroup = data_classGroup[1]
        else:
            candidate_classGroup = {}
            num_ = data_classGroup[4]['num']
            bin_ = data_classGroup[4]['bin']

            if num_ is None and bin_ is None:
                num_bin = None
            elif num_ is None:
                num_bin = bin_
            elif bin_ is None:
                num_bin = num_
            else:
                num_bin = num_ + bin_

            candidate_classGroup['num'] = data_classGroup[0][:, :num_]
            candidate_classGroup['bin'] = data_classGroup[0][:, num_:num_bin]
            candidate_classGroup['cat'] = data_classGroup[0][:, num_bin:].to(torch.int64)
            candidate_y_classgroup = data_classGroup[1]

        with torch.set_grad_enabled(
                torch.is_grad_enabled() and not self.memory_efficient
        ):
            # NOTE: during evaluation, candidate keys can be computed just once, which
            # looks like an easy opportunity for optimization. However:
            # - if your dataset is small or/and the encoder is just a linear layer
            #   (no embeddings and encoder_n_blocks=0), then encoding candidates
            #   is not a bottleneck.
            # - implementing this optimization makes the code complex and/or unobvious,
            #   because there are many things that should be taken into account:
            #     - is the input coming from the "train" part?
            #     - is self.training True or False?
            #     - is PyTorch autograd enabled?
            #     - is saving and loading checkpoints handled correctly?
            # This is why we do not implement this optimization.

            # When memory_efficient is True, this potentially heavy computation is
            # performed without gradients.
            # Later, it is recomputed with gradients only for the context objects.
            candidate_k = (
                self._encode(candidate_x_)[1]
                if self.candidate_encoding_batch_size is None
                else torch.cat(
                    [
                        self._encode(x)[1]
                        for x in delu.iter_batches(
                        candidate_x_, self.candidate_encoding_batch_size
                    )
                    ]
                )
            )
            ###group_feture encoding
            candidate_group_feture = (
                self._encode(candidate_classGroup)[1]
                if self.candidate_encoding_batch_size is None
                else torch.cat(
                    [
                        self._encode(x)[1]
                        for x in delu.iter_batches(
                        candidate_classGroup, self.candidate_encoding_batch_size
                    )
                    ]
                )
            )
        x, k = self._encode(x_)  # x一次线性层并变维；k经过两次线性层，第一次同x,第二次未变维
        if is_train:
            # NOTE: here, we add the training batch back to the candidates after the
            # function `apply_model` removed them. The further code relies
            # on the fact that the first batch_size candidates come from the
            # training batch.
            assert y is not None
            candidate_k = torch.cat([k, candidate_k])  # batch的x放在前batch size行
            candidate_y = torch.cat([y, candidate_y])
            candidate_rank = torch.cat([rank, candidate_rank]) #lzd
            candidate_y_classgroup = candidate_y_classgroup #lzd
        else:
            assert y is None

        '''ordinal compute'''
        class_loss,Ldt,yw_dict,rw_dict = ordinal_compute.class_distance_compute(candidate_k,candidate_rank,candidate_y)

        # >>>
        # The search below is optimized for larger datasets and is significantly faster
        # than the naive solution (keep autograd on + manually compute all pairwise
        # squared L2 distances + torch.topk).
        # For smaller datasets, however, the naive solution can actually be faster.
        batch_size, d_main = k.shape
        device = k.device
        with torch.no_grad():
            if self.search_index is None:
                self.search_index = (
                    faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d_main)
                    if device.type == 'cuda'
                    else faiss.IndexFlatL2(d_main)
                )
            # Updating the index is much faster than creating a new one.
            self.search_index.reset()
            self.search_index.add(candidate_k)  # type: ignore[code]
            distances: Tensor
            context_idx: Tensor
            distances, context_idx = self.search_index.search(  # type: ignore[code]
                k, context_size + (1 if is_train else 0)  # k:tensor(256,254)是x经过两次线性变换得到的encode,toml定义的96
            )  # 第一列0~255的x，后面96列是检索的idx

            ##lzd：增加class group 的检索
            self.search_index.reset()
            self.search_index.add(candidate_group_feture)  # type: ignore[code]
            distances_classGroup: Tensor
            context_idx_classGroup: Tensor
            distances_classGroup, context_idx_classGroup = self.search_index.search(k, 10)

            if is_train:
                # NOTE: to avoid leakage, the index i must be removed from the i-th row,
                # (because of how candidate_k is constructed).
                distances[
                    context_idx == torch.arange(batch_size, device=device)[:, None]
                    # ] = torch.inf#lzd：pytorch1.8无torch.inf
                    ] = float('inf')  # 排除x的idx
                # Not the most elegant solution to remove the argmax, but anyway.
                context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])  # argsort()升序后返回位置
                context_idx_classGroup = context_idx_classGroup.gather(-1, distances_classGroup.argsort())  # lzd

        if self.memory_efficient and torch.is_grad_enabled():
            assert is_train
            # Repeating the same computation,
            # but now only for the context objects and with autograd on.
            context_k = self._encode(
                {
                    ftype: torch.cat([x_[ftype], candidate_x_[ftype]])[
                        context_idx
                    ].flatten(0, 1)
                    for ftype in x_
                }
            )[1].reshape(batch_size, context_size, -1)
        else:
            context_k = candidate_k[context_idx]  # (batchsize,contextsize,dim)
            context_k_classGroup = candidate_group_feture[context_idx_classGroup]  ##lzd

        # In theory, when autograd is off, the distances obtained during the search
        # can be reused. However, this is not a bottleneck, so let's keep it simple
        # and use the same code to compute `similarities` during both
        # training and evaluation.

        # context_y = candidate_y[context_idx]
        # context_rank = candidate_rank[context_idx]  # lzd:获取retrieval的样本label
        '''-||k-ki||**2'''
        ##lzd增加实际距离和数据平衡参数

        context_y = candidate_y[context_idx]
        context_y = torch.tensor([[yw_dict[str(key)][0] for key in row] for row in context_y.tolist()]).to(device)##LZD:OKA
        similarities = -(
                k.square().sum(-1, keepdim=True)
                - (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
                + context_k.square().sum(-1)
        )  ##batch中每个xi与检索到的96个xii之间的相似度（batch szie ,96）
        # lzd=similarities*context_yw_r
        probs = F.softmax(similarities, dim=-1)  # (batchsize,contextsize)
        # probs = F.softmax(similarities, dim=-1)*context_yw_r
        probs = self.dropout(probs)  # 为何要dropout一些

        '''W(yi)+T(k-ki)'''
        # lzd=context_y[..., None]
        context_y_emb = self.label_encoder_(context_y[..., None].to(device))  # context_y_emb:(batchsize,contextsize,dmain)
        values = context_y_emb + self.T(k[:, None] - context_k)  ##(batchsize,contextsize,dmain)

        '''
        attention(Q,K,V)=softmax(Q,K/sqrt(d))*V --->attention(Q,K,V)=probs*values=softmax(-||k-ki||**2) * [W(yi)+T(k-ki)]
        此处即得到retrieval_x
        '''
        context_x = (probs[:, None] @ values).squeeze(1)  # (batchsize,dmain)
        '''x + retrieval_x后送入Predictor模块'''
        # x_bak = x
        x = x + context_x
        # x = torch.cat((x, context_x), dim=1) ##lzd-cat3

        # ##lzd：增加class group 的检索结果融合
        # context_y_classGroup=candidate_y_classgroup[context_idx_classGroup]
        # context_rw_classGroup = torch.tensor([[rw_dict[str(int(key))][0] for key in row] for row in context_y_classGroup.tolist()])
        #
        # similarities_classGroup = (
        #         -k.square().sum(-1, keepdim=True)
        #         + (2 * (k[..., None, :] @ context_k_classGroup.transpose(-1, -2))).squeeze(-2)
        #         - context_k_classGroup.square().sum(-1)
        # )  ##batch中每个xi与检索到的96个xii之间的相似度（batch szie ,96）
        # probs_classGroup = F.softmax(similarities_classGroup, dim=-1)  # (batchsize,contextsize)
        # # probs_classGroup = F.softmax(similarities_classGroup, dim=-1)*context_rw_r  # (batchsize,contextsize)
        # probs_classGroup = self.dropout(probs_classGroup)  # 为何要dropout一些
        #
        # context_y_emb_classGroup = self.label_encoder_(context_rw_classGroup[..., None].to(device))  # context_y_emb:(batchsize,contextsize,dmain)
        # values_classGroup = context_y_emb_classGroup + self.T(k[:, None] - context_k_classGroup)  ##(batchsize,contextsize,dmain)
        # context_x_classGroup = (probs_classGroup[:, None] @ values_classGroup).squeeze(1)  # (batchsize,dmain)

        # x = x + context_x_classGroup  # lzd-add
        # x = torch.cat((x, context_x_classGroup), dim=1)  ###lzd-cat1
        # x = torch.cat((x, x_bak+context_x_classGroup), dim=1)  ###lzd-cat2

        # >>>P 送入Predictor模块
        for block in self.blocks1:
            x = x + block(x)
        x = self.head(x)  ##（batchsize，1）此处得到是y_pred
        return x,class_loss.squeeze(),Ldt.squeeze()

def main(
        config: lib.JSONDict, output: Union[str, Path], *, force: bool = False
) -> Optional[lib.JSONDict]:
    # >>> start
    print('>>> tabr.py start')
    if not lib.start(output, force=force):
        return None

    output = Path(output)
    report = lib.create_report(config)
    C = lib.make_config(Config, config)

    delu.random.seed(C.seed)
    device = lib.get_device()

    # >>> data
    dataset = (
        C.data if isinstance(C.data, lib.Dataset) else lib.build_dataset(**C.data)
    ).to_torch(device)
    if dataset.is_regression:
        dataset.data['Y'] = {k: v.float() for k, v in dataset.Y.items()}
    Y_train = dataset.Y['train'].to(
        torch.long if dataset.is_multiclass else torch.float
    )
    # ###每类分成n小组
    data_classGroup = (split_class_group(dataset,3, device=device))  # lzd

    # >>> model
    model = Model(
        n_num_features=dataset.n_num_features,
        n_bin_features=dataset.n_bin_features,
        cat_cardinalities=dataset.cat_cardinalities(),
        n_classes=dataset.n_classes(),
        **C.model,
    )
    report['n_parameters'] = lib.get_n_parameters(model)  # 参数量
    logger.info(f'n_parameters = {report["n_parameters"]}')  # 打印参数量
    report['prediction_type'] = None if dataset.is_regression else 'logits'
    model.to(device)

    # if torch.cuda.device_count() > 1:
    #     print('torch.cuda.device_count()',torch.cuda.device_count())
    #     model = nn.DataParallel(model)  # type: ignore[code]

    # >>> training
    def zero_wd_condition(
            module_name: str,
            module: nn.Module,
            parameter_name: str,
            parameter: nn.parameter.Parameter,
    ):

        return (
                'label_encoder' in module_name
                or 'label_encoder' in parameter_name
                or lib.default_zero_weight_decay_condition(
            module_name, module, parameter_name, parameter
        )
        )

    optimizer = lib.make_optimizer(
        model, **C.optimizer, zero_weight_decay_condition=zero_wd_condition
    )
    loss_fn = lib.get_loss_fn(dataset.task_type)

    train_size = dataset.size('train')
    train_indices = torch.arange(train_size, device=device)

    epoch = 0
    eval_batch_size = 32768
    chunk_size = None
    progress = delu.ProgressTracker(C.patience)
    training_log = []
    writer = torch.utils.tensorboard.SummaryWriter(output)  # type: ignore[code]

    def get_Xy(part: str, idx) -> tuple[dict[str, Tensor], Tensor]:
        batch = (
            {
                key[2:]: dataset.data[key][part]
                for key in dataset.data
                if key.startswith('X_')
            },
            dataset.Y[part],
            dataset.data['Y_rank'][part]
        )
        return (
            batch
            if idx is None
            # else ({k: v[idx] for k, v in batch[0].items()}, batch[1][idx])
            else ({k: v[idx] for k, v in batch[0].items()}, batch[1][idx], batch[2][idx])
        )

    def apply_model(part: str, idx: Tensor, training: bool):
        # x, y = get_Xy(part, idx)
        x, y, rank = get_Xy(part, idx)
        candidate_indices = train_indices
        is_train = part == 'train'
        if is_train:
            # NOTE: here, the training batch is removed from the candidates.
            # It will be added back inside the model's forward pass.
            # candidate_indices = candidate_indices[~torch.isin(candidate_indices, idx)]#pytorch1.8无isin
            # lzd:去掉candidate_indices中idx的样本
            expanded_idx = idx.unsqueeze(0).expand(candidate_indices.size(0), -1)
            candidate_indices = candidate_indices[~torch.any(candidate_indices.unsqueeze(1) == expanded_idx, dim=1)]
        candidate_x, candidate_y,candidate_rank = get_Xy(
            'train',
            # This condition is here for historical reasons, it could be just
            # the unconditional `candidate_indices`.
            None if candidate_indices is train_indices else candidate_indices,
        )
        y_pred,class_loss,Ldt = model.forward(
            x_=x,
            y=y if is_train else None,
            rank=rank, #lzd
            candidate_x_=candidate_x,
            candidate_y=candidate_y,
            candidate_rank=candidate_rank, #lzd
            context_size=C.context_size,
            data_classGroup=data_classGroup,
            is_train=is_train,
            task_type=dataset.task_type,
        )

        return y_pred.squeeze(-1),class_loss,Ldt

    # @torch.inference_mode()
    @torch.no_grad()
    def evaluate(parts: list[str], eval_batch_size: int):
        model.eval()
        predictions = {}
        for part in parts:
            while eval_batch_size:
                try:
                    predictions[part] = (
                        torch.cat(
                            [
                                apply_model(part, idx, False)[0]
                                # apply_model(part, idx, False)
                                for idx in torch.arange(
                                dataset.size(part), device=device
                            ).split(eval_batch_size)
                            ]
                        )
                        .cpu()
                        .numpy()
                    )
                except RuntimeError as err:
                    if not lib.is_oom_exception(err):
                        raise
                    eval_batch_size //= 2
                    logger.warning(f'eval_batch_size = {eval_batch_size}')
                else:
                    break
            if not eval_batch_size:
                RuntimeError('Not enough memory even for eval_batch_size=1')
        metrics = (
            dataset.calculate_metrics(predictions, report['prediction_type'])
            if lib.are_valid_predictions(predictions)
            else {x: {'score': -999999.0} for x in predictions}
        )
        return metrics, predictions, eval_batch_size

    def save_checkpoint():
        lib.dump_checkpoint(
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'random_state': delu.random.get_state(),
                'progress': progress,
                'report': report,
                'timer': timer,
                'training_log': training_log,
            },
            output,
        )
        lib.dump_report(report, output)
        lib.backup_output(output)

    print()
    timer = lib.run_timer()
    while epoch < C.n_epochs:
        print(f'[...] {lib.try_get_relative_path(output)} | {timer}')

        model.train()
        epoch_losses = []
        for batch_idx in tqdm(
                lib.make_random_batches(train_size, C.batch_size, device),
                desc=f'Epoch {epoch}',
        ):
            y_pred, class_loss, Ldt=apply_model('train', batch_idx, True)
            loss, new_chunk_size = lib.train_step_lc(
                optimizer,
                lambda idx: loss_fn(y_pred, Y_train[idx]),
                batch_idx,
                chunk_size or C.batch_size,
                # class_loss+Ldt,
                Ldt,
            )
            epoch_losses.append(loss.detach())
            if new_chunk_size and new_chunk_size < (chunk_size or C.batch_size):
                chunk_size = new_chunk_size
                logger.warning(f'chunk_size = {chunk_size}')

        epoch_losses, mean_loss = lib.process_epoch_losses(epoch_losses)
        metrics, predictions, eval_batch_size = evaluate(
            ['val', 'test'], eval_batch_size
        )
        lib.print_metrics(mean_loss, metrics)
        training_log.append(
            {'epoch-losses': epoch_losses, 'metrics': metrics, 'time': timer()}
        )
        writer.add_scalars('loss', {'train': mean_loss}, epoch, timer())
        for part in metrics:
            writer.add_scalars('score', {part: metrics[part]['score']}, epoch, timer())

        progress.update(metrics['val']['score'])
        if progress.success:
            lib.celebrate()  # 打印梅花
            report['best_epoch'] = epoch
            report['metrics'] = metrics
            save_checkpoint()
            lib.dump_predictions(predictions, output)

        elif progress.fail or not lib.are_valid_predictions(predictions):
            break

        epoch += 1
        print()
    report['time'] = str(timer)
    print('tabr.py~epoch', epoch)

    # >>> finish
    model.load_state_dict(lib.load_checkpoint(output)['model'])
    report['metrics'], predictions, _ = evaluate(
        ['train', 'val', 'test'], eval_batch_size
    )
    report['chunk_size'] = chunk_size
    report['eval_batch_size'] = eval_batch_size
    lib.dump_predictions(predictions, output)
    lib.dump_summary(lib.summarize(report), output)
    save_checkpoint()
    lib.finish(output, report)
    return report


if __name__ == '__main__':
    lib.configure_libraries()
    lib.run_Function_cli(main)
