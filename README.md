TabCGOK: Intra-Class Groups Retrieval and Inter-Class Ordinal Knowledge Augmented Network for Ordinal Tabular Data Prediction（2024 ECAI）

## Abstract
Ordinal tabular data, with advantages of structured knowledge representation in tabular data and the characteristic of inter-class ranks, has drawn increasing attention. However, existing retrieval-based tabular deep learning methods designed primarily for classical tabular data pay less attention to ordinal tabular data. Ordinal knowledge of ordinal tabular data provides a more explicit objective for tabular ordinal classification by considering both classification and regression properties. Furthermore, these approaches overlook the significance of intra-class group features which can balance the retrieved probability of various sample size groups and capture shared knowledge among multiple samples within same group. In this work, we propose the Intra-Class Groups Retrieval and Inter-Class Ordinal Knowledge Augmented Network (TabCGOK) model for ordinal tabular data prediction, equipped with Intra-Class Groups Retrieval (CG) module and Inter-Class Ordinal Knowledge Augmented (OK) module. The CG module provides intra-class group features candidate set for subsequent retrieval operation. It divides each class into several groups, then extracts the representation of each group as intra-class group features. And the intra-class group features candidate set consists of all intra-class group features from each class. The OK module is designed to capture inter-class ordinal knowledge. It estimates the ordinal distances by calculating inter-class feature distances, which could correspond to the inter-class non-isometric nature of ordinal knowledge, and then aggregates the previous ordinal distances to clarify the containment relationship of ordinal knowledge. OK module utilizes the attention mechanism for fusing the captured ordinal knowledge to retrieved intra-class group features. Finally, TabCGOK integrates fused intra-class group features with sample level features for ordinal tabular data prediction. Extensive experiments on several ordinal tabular datasets demonstrate the effectiveness of our method.

## TabCGOK motivation
![motivation](./paper_image/motivation.png)
<!--<img src="./paper_image/ordinal_attribution.png" width="200" height="150"> <img src="./paper_image/class_group.png" width="200" height="150">-->

## TabCGOK Framework
![framework](./paper_image/framework.png)
Framework of our approach. TabCGOK retrieves group-level similar features (CG) and fuses them with inter-class ordinal knowledge augmentation weights (OK) to obtain similar group-level contextual features, which are then fused with sample-level similar features and sample features to obtain the final feature representation. $GA$ denotes group algorithm, $MP$ denotes mean-pooling, $W_x$ and $W_k$ denote the encoder, $R$ denotes the retriever, $Dis$ denotes the distance algorithm, Cum denotes the cumulative algorithm, $Va$ denotes the value algorithm, $Sim$ denotes the similarity algorithm, $P$ denotes the predictor, and $x_i$ denotes query sample which is a validation or test sample instance.

## Datasets 
The original datasets urls are in our paper. We download them, then updown them at [original url](https://drive.google.com/file/d/1N3uxh5iL8VA60zgzaFAUT6zsZ6Pkk6xh/view?usp=drive_link).

The pre-processed datasets are in [pre-processed url](https://drive.google.com/drive/folders/1dB4SWJEAfmcQjzEU4Cit5eXNGpz9ZsUg?usp=drive_link). There are instructions for preprocessing the two datasets in ```data_processing.docx```. The data preprocessing code is primarily in ```lib/data_preprocess.py```.

## Experimental Setup
python=3.9, pytorch=1.12, 32G NVIDIA V100 GPU, 12G Tesla K80 GPUs, CPU. The version of each function package is shown in ``` environment.yaml```.

## Files illustration
'bin': contains baselines models and our model TabCGOK, as well as entry code for training, testing.

'data': contains pre-processed datasets. Since 7 datasets are too big, so here we only put one example dataset, the others can be downloaded from 'Dataset'.

'lib': contains some important functions of the models, especially our proposed CG module code splitClassGroup.py and OK module code ordinal_compute.py. It also contains the result calculation code compute_scoreMeanStd.py.

'exp': contains the experimental configuration of each model for each dataset, which contains the evaluation of ACC and RMSE, and the corresponding ```checkpiont.pt``` and results are saved in this folder after the model training and testing are completed.
## Training and testing
Building a conda environment, sometimes packages in ```environment.yaml``` need to be manually installed separately.

```
conda create -n TabCGOK python=3.9
conda activate TabCGOK
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r environment.yaml
```
Download the pre-processed datasets from [GoogleDrive](https://drive.google.com/drive/folders/1dB4SWJEAfmcQjzEU4Cit5eXNGpz9ZsUg?usp=drive_link), unzip it and put it in the ```data``` file. 

Let's take the abalone dataset as an example, and do the same with the other datasets.
Creating hyperparameter files
```
cd exp/TabCGOK/acc/abalone
cp 310abalone-tuning.toml abalone-tuning.toml
```
In fact, abalone-tuning.toml is an arbitrary name and you can choose a different one, but it must end with -tuning.toml.

Our code is written roughly, so there are two key hyperparameters that need to be set manually by going into the model code:

(1) Number of groups within the class: in code ```bin/TabCGOK.py```, change the ```5``` in this line of code```data_classGroup = (split_class_group(dataset,6, device=device))``` to the number of groups we want to have (e.g. 3),

```
data_classGroup = (split_class_group(dataset, 3, device=device))
```
(2) Number of group feature retrievals: in code ```bin/TabCGOK.py```, change the ```10``` in this line of code```distances_classGroup, context_idx_classGroup = self.search_index.search(k, 12)``` to the number of groups we want to have (e.g. 10),

```
distances_classGroup, context_idx_classGroup = self.search_index.search(k, 10)
```
(3) Setting the GPU number, in ```bin/go.py```, ```evaluate.py```, and ```tune.py```. It's best to use the same GPU number.
```
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
```
Note that 0 can be set depending on the server.

(4) Fine-tuning or direct training validation
(4.1) If you want to fine-tune, run the following:
```
python bin/tune.py exp/TabCGOK/acc/abalone/abalone-tuning.toml
```
Once the run is finished, the file ```exp/TabCGOK/acc/abalone/abalone-tuning``` should appear.

(4.2) If you want to direct training validation, run the following:
```
python bin/go.py exp/TabCGOK/acc/abalone/abalone-tuning.toml
```
Once the run is finished, the following directories should appear:
- `exp/TabCGOK/acc/abalone/abalone-tuning`
- `exp/TabCGOK/acc/abalone/abalone-evaluation`
- `exp/TabCGOK/acc/abalone/abalone-ensemble-5`

Note: If you need to re-run, you will need to delete the existing folder, e.g., if you have run (4.1) and then want to run (4.2), you will need to delete ```exp/TabCGOK/acc/abalone/abalone-tuning```, because the process of running (4.2) includes (4.1).

(5) Calculate the experimental results: in code ```lib/compute_scoreMeanStd.py```, Set the path in the ```Path(xxx)```. For example:

line 7: Getting fine-tuned evaluations.

```
    for x in Path('../exp/TabCGOK/acc/abalone/abalone-evaluation').iterdir()
```
        
line 15: Getting training validation evaluations.
```
for x in Path('../exp/TabCGOK/acc/abalone/abalone-ensemble-5').iterdir()
```

Run this to get results:
```
python lib/compute_scoreMeanStd.py
```
(6) The checkpiont and results of our TabCGOK model are available in [GoogleDrive](https://drive.google.com/file/d/1WrJ9a01waFQrag97l6tBLns-t_hT15lo/view?usp=drive_link).

(7) The ablation study code is in ```bin/tabCGOK_ICGR.py``` and ```bin/tabCGOK_OKA.py```. The confusion matrix code is in ```lib/confusion_matrix.py```.
## Citation(Accepted)
<!--
```bibtex
@InProceedings{Luo2024TabCGOK,
  author  = {Zhengdong Luo, Abibulla Atawulla, Fengyi Yang, Yongqing Zhu, Yixiao Ren, Yunfei Han and Xi Zhou},
  title   = {TabCGOK: Intra-Class Groups Retrieval and Inter-Class Ordinal Knowledge Augmented Network for Ordinal Tabular Data Prediction},
  booktitle = {ECAI},
  year    = {2024}
```
-->
```TabCGOK: Intra-Class Groups Retrieval and Inter-Class Ordinal Knowledge Augmented Network for Ordinal Tabular Data Prediction```
## Acknowledgements
Our code is based on [TabR](https://github.com/yandex-research/tabular-dl-tabr) and [OrdinalEntropy](https://github.com/needylove/OrdinalEntropy). Thanks for their great works!
