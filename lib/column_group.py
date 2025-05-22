from sklearn.cluster import KMeans
from collections import defaultdict
import torch
import numpy as np


from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import DBSCAN
def K_means(X_train,X_valid,X_test,k):
    data_x = torch.cat((X_train,X_valid,X_test), dim=0)
    columns_for_clustering= [i for i in range(data_x.shape[1])]##获取列号
    X = data_x[:, columns_for_clustering].cpu().numpy().astype(np.float32)
    kmeans = KMeans(n_clusters=k)# 使用K-means聚类算法进行聚类
    kmeans.fit(X.T)  # 对列进行聚类，需要转置数据
    labels = kmeans.labels_# 获取每个列所属的聚类簇
    clusters = defaultdict(list)# 创建字典记录每个簇中的列号
    for i, label in enumerate(labels):
        clusters[label].append(i)
    merged_clusters = defaultdict(list)# 合并只包含一个元素的簇为一类
    for cluster, column_indices in clusters.items():
        if len(column_indices) == 1:
            merged_clusters['merged'].extend(column_indices)
        else:
            merged_clusters[cluster].extend(column_indices)

    ##lzd:离散性--组合
    keys = list(merged_clusters.keys())
    num_keys = len(merged_clusters)
    group_discrete= {}
    for i in range(num_keys):
        key=f'X_discreteGroup_{i}'
        group_discrete[key]={}
        group_discrete[key]['train'] = X_train[:, merged_clusters[keys[i]]]
        group_discrete[key]['val'] = X_valid[:, merged_clusters[keys[i]]]
        group_discrete[key]['test'] = X_test[:, merged_clusters[keys[i]]]
    return group_discrete

def hierarchy_cluster(X_train,X_valid,X_test,k):
    data_x = torch.cat((X_train, X_valid, X_test), dim=0)
    columns_for_clustering= [i for i in range(data_x.shape[1])]##获取列号
    X = data_x[:, columns_for_clustering].cpu().numpy().astype(np.float32)
    # 计算距离矩阵
    dist_matrix = linkage(X.T, method='complete', metric='euclidean')

    # 将数据点分配到组中
    clusters = fcluster(dist_matrix, k, criterion='maxclust')

    # 创建字典保存每个组包含的列名
    group_columns = {}
    unique_clusters = np.unique(clusters)
    for cluster_id in unique_clusters:
        columns_in_cluster = np.where(clusters == cluster_id)[0]
        group_columns[cluster_id] = columns_in_cluster
    return group_columns
