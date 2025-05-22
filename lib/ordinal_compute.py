# coding=utf-8

import faiss
import torch
import torch.nn.functional as F
from scipy.spatial import distance
from collections import defaultdict

def class_rank_loss(class_rank):
    loss = 0
    for i in range(1, len(class_rank)):
        diff = class_rank[i] - class_rank[i-1]
        loss += F.relu(-diff)  # 使用负值作为损失，确保 class_rank 是升序形式
    return loss
def Divide_category_objects(y):  ##各类样本按类划分 Divide x samples(objects) by category
    category_index =defaultdict(list)
    for i in range(len(y)):
        label = str(y[i].item())
        object_index = i
        category_index[label].append(object_index)
    return category_index


def compute_similarity(k, context_k, dropout):
    similarities = (
            -k.square().sum(-1, keepdim=True)
            + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
            - context_k.square().sum(-1)
    )
    probs = F.softmax(similarities, dim=-1)  #
    probs = dropout(probs)  #
    return probs


def compute_values(k, context_k, context_y, label_encoder, T):
    context_y_emb = label_encoder(context_y[..., None])  # context_y_emb:(batchsize,contextsize,dmain)
    values = context_y_emb + T(k[:, None] - context_k)  ##(batchsize,contextsize,dmain)
    return values


def compute_search_index(k, candidate_k, context_size, search_index):
    device = k.device
    dim = k.size(1)
    with torch.no_grad():
        if device.type == 'cuda':
            search_index = (faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), dim))
        else:
            faiss.IndexFlatL2(dim)
        # Updating the index is much faster than creating a new one.
        search_index.reset()
        search_index.add(candidate_k)
        k = k.contiguous()
        distances, context_idx = search_index.search(k, context_size)
    return distances, context_idx


def retrieval_module(anchor_object, candiate_object, context_size):
    device = anchor_object.device
    with torch.no_grad():
        d = anchor_object.shape  ## 获取特征维度
        if device.type == 'cuda':  # 根据设备类型选择Faiss索引类型
            index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d)
        else:
            index = faiss.IndexFlatL2(d)

        index.reset()  # 清空索引并重新初始化，以确保索引处于空白状态
        index.add(candiate_object)  # 将候选对象添加到索引中
        distances, indices = index.search(anchor_object, context_size)  # 通过索引进行检索

    return distances, indices

def compute_neighborCatagory(trainY):
    '''#对train x:按类划分'''
    Divide_trainX_dict = Divide_category_objects(trainY) ##label:{indexs}
    '''#对category : 计算相邻类'''# 构建训练集相邻候选类
    train_candidate_y = {}
    train_categories = torch.unique(trainY)  # 数据集包含的类别
    for i in range(len(train_categories)):
        # 获取左右类的label
        y_left = train_categories[i - 1].item() if i - 1 >= 0 else None
        y_right = train_categories[i + 1].item() if i + 1 < len(train_categories) else None
        # 获取左右类的object
        object_left = Divide_trainX_dict[str(y_left)] if y_left is not None else None
        object_left_label = torch.full((len(Divide_trainX_dict[str(y_left)]),), y_left) if y_left is not None else None
        object_right = Divide_trainX_dict[str(y_right)] if y_right is not None else None
        object_right_label = torch.full((len(Divide_trainX_dict[str(y_right)]),), y_right) if y_right is not None else None
        # 构建候选字典
        if object_left is not None and object_right is not None:
            train_candidate_y[str(train_categories[i].item())] = torch.cat((object_left_label, object_right_label), dim=0)
        elif object_left is not None:
            train_candidate_y[str(train_categories[i].item())] = object_left_label
        elif object_right is not None:
            train_candidate_y[str(train_categories[i].item())] = object_right_label
        else:
            train_candidate_y[str(train_categories[i].item())] = None
    return train_candidate_y
def retrieval_NeighboringCategory(context_k,context_y,context_rank, device):

    # (2)ordinal retrieval 的输出为(96*context_size,x_dim),然后加入context_k中，即对context_k扩容，context_x=cat(context_x,ordinal_context_x)
    # (1)ordinal retrieval 的输出为(context_size,x_dim)，然后并行的与tabr context_k进行相同操作，即x=x+context_x+ordinal_context_x(再次检索复杂度很高，要针对batch的每个x检索)
    # 方案(1)从相似度检索结果中继续使用rank信息进行精细筛选
    # train_candidate_x, train_candidate_y=compute_neighborCatagory(trainX,trainY)
    # dict_category_i=defaultdict(list)
    context_k_rank={}
    context_y_rank = {}
    for i in range(context_k.size(0)):  # X:tensor(batchsize,96,dim)
        context_ki = context_k[i:i + 1].squeeze(0)  # tensor(1,96,dim)
        context_yi = context_y[i:i + 1].squeeze(0)
        context_ranki=context_rank[i:i + 1].squeeze(0)
        '''按类划分，形成成批检索，加快检索速度'''
        Divide_yX_index = Divide_category_objects(context_ranki)
        '''#根据X中category,从相应的相邻类中检索候选集'''
        most_key = max(Divide_yX_index, key=lambda k: len(Divide_yX_index[k]))
        _left=str(float(most_key)-1)
        _right=str(float(most_key)+1)

        most_key_index=Divide_yX_index[most_key]
        key_most=context_yi[most_key_index]

        values_left=context_ki[Divide_yX_index[_left]] if _left in Divide_yX_index.keys() else None
        key_lift=context_yi[Divide_yX_index[_left]] if _left in Divide_yX_index.keys() else None

        values_right=context_ki[Divide_yX_index[_right]] if _right in Divide_yX_index.keys() else None
        key_right=context_yi[Divide_yX_index[_right]] if _right in Divide_yX_index.keys() else None

        if values_left is not None and values_right is not None:
            values_x=torch.cat((values_left,context_ki[most_key_index],values_right),dim=0)
            values_y = torch.cat((key_lift,key_most,key_right),dim=0)
        elif values_left is not None:
            values_x = torch.cat((values_left, context_ki[most_key_index]), dim=0)
            values_y = torch.cat((key_lift,key_most),dim=0)
        elif values_right is not None:
            values_x = torch.cat((context_ki[most_key_index], values_right), dim=0)
            values_y = torch.cat((key_most,key_right),dim=0)
        else:
            values_x=context_ki[most_key_index]
            values_y = key_most
        context_k_rank[i]=values_x
        context_y_rank[i]=values_y
    return context_k_rank,context_y_rank

'''
计算类间实际距离，取代模型参数中对y的使用
y=c_dist[0]
'''
def class_distance_compute(x, rank,y):
    f_n, f_c = x.size()
    u_value, u_index, u_counts = torch.unique(rank, return_inverse=True, return_counts=True)  # 集合,挑出tensor中的独立不重复元素,
    v_value, v_index, v_counts = torch.unique(y, return_inverse=True, return_counts=True)
    center_f = torch.zeros([len(u_value), f_c]).cpu()
    u_index = u_index.squeeze()
    center_f.index_add_(0, u_index.cpu(), x.cpu())  # 相同类的值加起来
    u_reciprocal = 1 / u_counts
    u_counts = u_counts.unsqueeze(1)
    center_f = center_f.cpu() / u_counts.cpu()  # 类均值特征

    center_fn= F.normalize(center_f, dim=1)
    _distance = distance.cdist(center_fn.cpu().detach().numpy(), center_fn.cpu().detach().numpy(), metric='minkowski', p=2)#p=1,2,float('inf')
    class_distance_tensor = torch.from_numpy(_distance).cpu()

    class_rank=class_distance_tensor[0].unsqueeze(1).to(x.device)#取第一行，各类与第一类的距离 #tensor([[0.0000], [0.5790], [0.4820], [0.9720],[1.3143], [1.4088],[1.3611]])

    # class_rank=torch.diagonal(class_distance_tensor, offset=1).unsqueeze(1).to(device=x.device)## 计算相邻类的距离
    # class_rank= torch.cat((torch.tensor([[0]]).to(device=x.device),class_rank), dim=0)

    class_loss=class_rank_loss(class_rank) ##确保rank为升序
    class_rank = class_rank.cumsum(dim=0) ##tensor([[0.0000], [0.5790], [1.0609],[2.0330],[3.3472], [4.7560],[6.1171]])
    y_value = class_rank+ u_value[0]  ##实际等级值

    y_weight = up_triu(euclidean_dist(y_value, y_value))
    y_weight = y_weight.cumsum(dim=0)
    y_max = torch.max(y_weight)
    y_min = torch.min(y_weight)
    _weight = ((y_weight - y_min) / y_max)#标准化

    '''构建y-rank字典'''
    mean = y_value.mean()
    std = y_value.std()
    normalized_y_value = (y_value - mean) / std   #normalized_y_value=tensor([[-1.6268], [-0.5490],[-0.7296], [ 0.1827], [ 0.8198],[ 0.9958],[ 0.9071]])
    ##normalized_y_value.cum=tensor([[-1.1228],[-0.8685],[-0.6568],[-0.2298],[ 0.3475],[ 0.9663],[ 1.5641]])

    yw_dict = {str(key): (value[0], value_) for key, value, value_ in zip(v_value.tolist(), normalized_y_value.tolist(), u_reciprocal.tolist())}
    # rw_dict = {str(key): (value[0], value_, value_u) for key, value, value_ ,value_u in zip(u_value.tolist(), normalized_y_value.tolist(), u_reciprocal.tolist(), v_value.tolist())}
    rw_dict = {str(int(key)): (value[0], value_, value_u) for key, value, value_ ,value_u in zip(u_value.tolist(), normalized_y_value.tolist(), u_reciprocal.tolist(), v_value.tolist())}
    """L_d = - mean(w_ij ||z_{c_i} - z_{c_j}||_2)"""
    _distance = torch.from_numpy(_distance)
    _distance = up_triu(_distance).to(x.device) * _weight.to(x.device)
    L_d = - torch.mean(_distance)

    """L_t = - mean( ||z_{c_i} - z_{c_j}||_2)"""
    _features = F.normalize(x, dim=1)
    # _features_center = center_fn[u_index, :].to(x.device)
    _features_center = center_fn.to(x.device)[u_index.to(x.device), :].to(x.device)
    _features = _features - _features_center
    _features = _features.pow(2)
    _tightness = torch.sum(_features, dim=1)
    _mask = _tightness > 0
    _tightness = _tightness[_mask]
    L_t = torch.mean(_tightness)

    Ldt=L_t #ordinal entropy

    # class_rank_cumsum = {index: value.item() for index, value in enumerate(class_rank.cumsum(dim=0))}
    # y_weight_cumsum={index: value.item() for index, value in enumerate(y_weight[0:len(u_value)])}
    return class_loss, Ldt, yw_dict, rw_dict

def up_triu(x):
    # return a flattened view of up triangular elements of a square matrix
    n, m = x.shape
    assert n == m
    _tmp = torch.triu(torch.ones(n, n), diagonal=1).to(torch.bool) #上三角矩阵部分
    return x[_tmp]
def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n) #平方和
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t()) ##Euclidean distance: under the 2nd root sign (xx + yy - 2xy)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist
# def rerank(query, search_results):
#   search_results["old_similarity_rank"] = search_results.index+1 # Old ranks
#
#   torch.cuda.empty_cache()
#   gc.collect()
#
#   search_results["new_scores"] = reranker_model.compute_score([[query,chunk] for chunk in search_results["text"]]) # Re compute ranks
#   return search_results.sort_values(by = "new_scores", ascending = False).reset_index(drop = True)

