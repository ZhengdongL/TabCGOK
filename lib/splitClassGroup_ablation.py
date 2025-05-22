from collections import defaultdict

import numpy as np
import torch
from sklearn.cluster import KMeans


def K_means(x,k):
    kmeans = KMeans(n_clusters=k,n_init=10)
    kmeans.fit(x.cpu())
    labels = kmeans.labels_# 获取每个列所属的聚类簇
    clusters = defaultdict(list)# 创建字典记录每个簇中的列号
    for i, label in enumerate(labels):
        clusters[label].append(i)

    group_features = []
    for key,cluster_index in clusters.items():
        selected_features = x[cluster_index]
        mean_feature = torch.mean(selected_features, dim=0)
        group_features.append(mean_feature)
    group_features = torch.stack(group_features)
    return group_features

def split_class_group(dataset,cluster_num,device):
    X_num_train = dataset.data['X_num']['train'] if dataset.X_num is not None else torch.Tensor().to(device)
    X_bin_train = dataset.data['X_bin']['train'] if dataset.X_bin is not None else torch.Tensor().to(device)
    X_cat_train = dataset.data['X_cat']['train'] if dataset.X_cat is not None else torch.Tensor().to(device)

    num_c=X_num_train.shape[1] if X_num_train.dim() >1 else None
    bin_c = X_bin_train.shape[1] if X_bin_train.dim() >1 else None
    cat_c = X_cat_train.shape[1] if X_cat_train.dim() >1 else None
    dim_dict= {'num':num_c,'bin':bin_c,'cat':cat_c}  # 存储各类型的的列数

    features = torch.cat((X_num_train, X_bin_train, X_cat_train), dim=1)
    y_gt = dataset.data['Y_rank']['train'] if dataset.Y['train'] is not None else torch.Tensor().to(device)

    f_n, f_c = features.size()
    u_value, u_index, u_counts = torch.unique(y_gt, return_inverse=True, return_counts=True)

    center_f = torch.zeros([len(u_value), f_c]).cpu()
    u_index = u_index.squeeze()
    center_f.index_add_(0, u_index.cpu(), features.cpu())  # 相同类的值加起来
    u_counts = u_counts.unsqueeze(1)
    center_f = center_f.cpu() / u_counts.cpu()  # 类均值特征

    class_cf=center_f.to(features.device)
    class_y=u_value.to(features.device)

    index_dict = {}
    for i, label in enumerate(u_value):
        label_indices = torch.nonzero(u_index == i).view(-1)
        index_dict[label.item()] = label_indices.tolist()
    mini_group_features = torch.zeros(1, f_c)
    mini_group_y = torch.zeros(1, )
    for y,index in index_dict.items():
        x=features[index]
        group_features=K_means(x,cluster_num)
        y_tensor=y*torch.ones(cluster_num)
        mini_group_features=torch.cat((mini_group_features.to(device),group_features.to(device)),dim=0)
        mini_group_y=torch.cat((mini_group_y.to(device),y_tensor.to(device)),dim=0)
    return mini_group_features[1:], mini_group_y[1:], class_cf, class_y,dim_dict

