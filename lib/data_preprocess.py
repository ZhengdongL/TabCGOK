# coding=utf-8
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
import json
import os
from collections import defaultdict
from scipy.io import arff
from column_group import K_means
import torch
from io import StringIO
'''
https://github.com/yandex-research/tabular-dl-tabr
Let's say your dataset is called my-dataset. Then, create the data/my-dataset directory with the following content:

If the dataset has numerical (i.e. continuous) features
    Files: X_num_train.npy, X_num_val.npy, X_num_test.npy
    NumPy data type: np.float32
If the dataset has binary features
    Files: X_bin_train.npy, X_bin_val.npy, X_bin_test.npy
    NumPy data type: np.float32
    All values must be 0.0 and 1.0
If the dataset has categorical features
    Files: X_cat_train.npy, X_cat_val.npy, X_cat_test.npy
    NumPy data type: np.str_ (yes, the values must be strings)
Labels
    Files: Y_train.npy, Y_val.npy, Y_test.npy
    NumPy data type: np.float32 for regression, np.int64 for classification
    For classification problems, the labels must from the range [0, ..., n_classes - 1].
info.json -- a JSON file with the following keys:
    "task_type": one of "regression", "binclass", "multiclass"
    (optional) "name": any string (a "pretty" name for your dataset, e.g. "My Dataset")
    (optional) "id": any string (must be unique among all "id" keys of all info.json files of all datasets in data/)
    
    eg:info.json
    {
        "name": "Adult",
        "id": "adult--default",
        "task_type": "binclass",
        "n_num_features": 6,
        "n_bin_features": 1,
        "n_cat_features": 7,
        "test_size": 16281,
        "train_size": 26048,
        "val_size": 6513
    }
READY -- just an empty file
'''
def convert_to_numeric(value):
    try:
        if value is None:
            return 2024 #随意设置为某值
        return float(value)
    except ValueError:
        return value

# def normalize_reg_label(label, y_mu_std):
#     mu, std = y_mu_std.astype(float).mean(), y_mu_std.astype(float).std()
#     norm_label = ((label.astype(np.float32) - mu) / std)
#     norm_label = norm_label.reshape(-1, 1)
#     return norm_label


# def quantile_transform(X_train, X_valid, X_test):
#     quantile_train = np.copy(X_train)
#     qt = QuantileTransformer(random_state=55688, output_distribution='normal').fit(quantile_train)
#     X_train = qt.transform(X_train)
#     X_valid = qt.transform(X_valid)
#     X_test = qt.transform(X_test)
#     return X_train, X_valid, X_test


def identify_num_cat_bin_feature(unique_values):
    # if len(unique_values) <= 10 and len(unique_values)>=3:
    if isinstance(unique_values[0], (int, float)):
        identify = 'numerical'
    elif len(unique_values) == 2:
        identify = 'binary'
    else:
        identify = 'numerical'
    return identify


##依照值的类型
def get_feature_types(dataframe):
    column_types = dataframe.dtypes
    numerical_features = dataframe.select_dtypes(include=[np.number]).columns
    binary_features = [col for col in dataframe.columns if dataframe[col].nunique() == 2 and col in numerical_features]
    # binary_features = [col for col in dataframe.columns if dataframe[col].nunique() == 2 and set(dataframe[col].unique()).issubset({0, 1})] #All values must be 0.0 and 1.0
    categorical_features = dataframe.select_dtypes(include=[object]).columns # the values must be strings
    # categorical_features = dataframe.select_dtypes(include=[pd.StringDtype]).columns
    # 从 numerical_features 中排除 binary_features 和 categorical_features
    numerical_features = numerical_features.difference(binary_features).difference(categorical_features)

    return numerical_features, binary_features, categorical_features
##依照值个数
# def get_feature_types(dataframe):
#     numerical_features = []
#     binary_features = []
#     categorical_features = []
#     for column in dataframe.columns:
#         unique_values = dataframe[column].nunique()
#         if unique_values == 2:
#             binary_features.append(column)
#         elif unique_values >10:
#             numerical_features.append(column)
#         else:
#             categorical_features.append(column)
#     return numerical_features, binary_features, categorical_features


def preprocess_yahoo_regression(train_path, valid_path, test_path,task_type):
    for fname in (train_path, valid_path, test_path):
        raw = open(fname).read().replace('\\t', '\t')
        with open(fname, 'w') as f:
            f.write(raw)

    data_train = pd.read_csv(train_path, header=None, skiprows=1, sep='\t')
    data_valid = pd.read_csv(valid_path, header=None, skiprows=1, sep='\t')
    data_test = pd.read_csv(test_path, header=None, skiprows=1, sep='\t')

    # with open(train_path, 'r') as file:
    #     train_content = file.read().replace('\\t', '\t').split('\n')
    # with open(valid_path, 'r') as file:
    #     valid_content = file.read().replace('\\t', '\t').split('\n')
    # with open(test_path, 'r') as file:
    #     test_content = file.read().replace('\\t', '\t').split('\n')
    # data_train=pd.DataFrame(train_content).iloc[1:, 0].str.split('\t', expand=True).map(convert_to_numeric)
    # data_valid=pd.DataFrame(valid_content).iloc[1:, 0].str.split('\t', expand=True).map(convert_to_numeric)
    # df_test= pd.DataFrame(test_content).iloc[1:, 0].str.split('\t', expand=True).map(convert_to_numeric)

    X_train, y_train, query_train = data_train.iloc[:, 2:].values, data_train.iloc[:, 0].values, data_train.iloc[:, 1].values
    X_valid, y_valid, query_valid = data_valid.iloc[:, 2:].values, data_valid.iloc[:, 0].values, data_valid.iloc[:, 1].values
    X_test, y_test, query_test = data_test.iloc[:, 2:].values, data_test.iloc[:, 0].values, data_test.iloc[:, 1].values
    # X_train, X_valid, X_test = quantile_transform(X_train, X_valid, X_test)
    u_value, u_index, u_counts = torch.unique(torch.tensor(y_train), return_inverse=True, return_counts=True)
    print(u_value)
    print(u_counts)

    df_x = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)], axis=0)  ## Concatenate train, valid, and test dataframes
    df_x = df_x.map(convert_to_numeric) ##将DataFrame中的数据转换为数值型，但保留无法转换的值为字符串类型
    column_types = df_x.dtypes
    numerical_features, binary_features, categorical_features = get_feature_types(df_x)## Get feature types
    # Get indices of feature types
    numerical_feature_indices = [df_x.columns.get_loc(feature) for feature in numerical_features]
    binary_feature_indices = [df_x.columns.get_loc(feature) for feature in binary_features]
    categorical_feature_indices = [df_x.columns.get_loc(feature) for feature in categorical_features]


    ##train set
    X_num_train = pd.DataFrame(X_train[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_train is not None and np.save(os.path.join(save_path, "X_num_train.npy"), X_num_train)  # Numerical features
    X_bin_train = pd.DataFrame(X_train[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_train is not None and np.save(os.path.join(save_path, "X_bin_train.npy"), X_bin_train)  # Binary features
    X_cat_train = pd.DataFrame(X_train[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_train is not None and np.save(os.path.join(save_path, "X_cat_train.npy"), X_cat_train)  # Categorical features
    # Y_train = y_train  ##因为要在后续程序中用到原始class rank
    Y_train = y_train.astype(np.float32) if task_type == "regression" else y_train.astype(np.int64)
    U_value, V_index, U_counts = torch.unique(torch.tensor(Y_train), return_inverse=True, return_counts=True)
    print('U_value', U_value)
    print('U_counts', U_counts)
    np.save(os.path.join(save_path, "Y_train.npy"), Y_train)  # Labels

    ##validation set
    X_num_valid = pd.DataFrame(X_valid[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_valid is not None and np.save(os.path.join(save_path, "X_num_val.npy"), X_num_valid)  # Numerical features
    X_bin_valid = pd.DataFrame(X_valid[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_valid is not None and np.save(os.path.join(save_path, "X_bin_val.npy"), X_bin_valid)  # Binary features
    X_cat_valid = pd.DataFrame(X_valid[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_valid is not None and np.save(os.path.join(save_path, "X_cat_val.npy"), X_cat_valid)  # Categorical features
    # Y_valid = y_valid  ##因为要在后续程序中用到原始class rank
    Y_valid = y_valid.astype(np.float32) if task_type == "regression" else y_valid.astype(np.int64)
    U_value, V_index, U_counts = torch.unique(torch.tensor(Y_valid), return_inverse=True, return_counts=True)
    print('U_value', U_value)
    print('U_counts', U_counts)
    np.save(os.path.join(save_path, "Y_val.npy"), Y_valid)  # Labels

    ##test set
    X_num_test = pd.DataFrame(X_test[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_test is not None and np.save(os.path.join(save_path, "X_num_test.npy"), X_num_test)  # Numerical features
    X_bin_test = pd.DataFrame(X_test[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_test is not None and np.save(os.path.join(save_path, "X_bin_test.npy"), X_bin_test)  # Binary features
    X_cat_test = pd.DataFrame(X_test[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_test is not None and np.save(os.path.join(save_path, "X_cat_test.npy"), X_cat_test)  # Categorical features
    # Y_test = normalize_reg_label(y_test,y_train) if task_type == "regression" else y_test.astype(np.int64)
    Y_test = y_test.astype(np.float32) if task_type == "regression" else y_test.astype(np.int64)
    U_value, V_index, U_counts = torch.unique(torch.tensor(Y_test), return_inverse=True, return_counts=True)
    print('U_value', U_value)
    print('U_counts', U_counts)
    np.save(os.path.join(save_path, "Y_test.npy"), Y_test)  # Labels


    # info.json
    info = {
        "name": 'yahoo',
        "id": "yahoo--s709877c5",
        "task_type": task_type,
        "n_num_features": len(numerical_feature_indices),
        "n_bin_features": len(binary_feature_indices),
        "n_cat_features": len(categorical_feature_indices),
        "train_size": X_train.shape[0],
        "val_size": X_valid.shape[0],
        "test_size": X_test.shape[0]
    }
    with open(os.path.join(save_path, "info.json"), "w") as f:
        json.dump(info, f, indent=4)


def preprocess_microsoft_regression(train_path, valid_path, test_path, task_type):
    print('\n process train')
    fidR = open(train_path, 'r')
    train_features = defaultdict(list)
    for fLine in fidR.readlines():
        elements = fLine.strip().split(' ')
        train_features[0].append(elements[0])
        for item in elements[1:]:
            if not item.startswith("qid:"):
                feature_id, feature_value = item.split(":")
                train_features[int(feature_id)].append(float(feature_value))

    print('process valid')
    fidR = open(valid_path, 'r')
    valid_features = defaultdict(list)
    for fLine in fidR.readlines():
        elements = fLine.strip().split(' ')
        valid_features[0].append(elements[0])
        for item in elements[1:]:
            if not item.startswith("qid:"):
                feature_id, feature_value = item.split(":")
                valid_features[int(feature_id)].append(float(feature_value))

    print('process test')
    fidR = open(test_path, 'r')
    test_features = defaultdict(list)
    for fLine in fidR.readlines():
        elements = fLine.strip().split(' ')
        test_features[0].append(elements[0])
        for item in elements[1:]:
            if not item.startswith("qid:"):
                feature_id, feature_value = item.split(":")
                test_features[int(feature_id)].append(float(feature_value))

    df_train = pd.DataFrame.from_dict(train_features, orient='columns').map(convert_to_numeric)
    df_valid = pd.DataFrame.from_dict(valid_features, orient='columns').map(convert_to_numeric)
    df_test = pd.DataFrame.from_dict(test_features, orient='columns').map(convert_to_numeric)

    X_train, y_train = df_train.iloc[:, 1:].values, df_train.iloc[:, 0].values
    X_valid, y_valid = df_valid.iloc[:, 1:].values, df_valid.iloc[:, 0].values
    X_test, y_test = df_test.iloc[:, 1:].values, df_test.iloc[:, 0].values
    # X_train, X_valid, X_test = quantile_transform(X_train, X_valid, X_test)
    u_value, u_index, u_counts = torch.unique(torch.tensor(y_train.astype(float)), return_inverse=True, return_counts=True)
    print(u_value)
    print(u_counts)
    df_x = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)], axis=0)  ## Concatenate train, valid, and test dataframes
    df_x = df_x.map(convert_to_numeric) ##将DataFrame中的数据转换为数值型，但保留无法转换的值为字符串类型
    column_types = df_x.dtypes
    numerical_features, binary_features, categorical_features = get_feature_types(df_x)  ## Get feature types
    # Get indices of feature types
    numerical_feature_indices = [df_x.columns.get_loc(feature) for feature in numerical_features]
    binary_feature_indices = [df_x.columns.get_loc(feature) for feature in binary_features]
    categorical_feature_indices = [df_x.columns.get_loc(feature) for feature in categorical_features]

    ##train set
    X_num_train = pd.DataFrame(X_train[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_train is not None and np.save(os.path.join(save_path, "X_num_train.npy"), X_num_train)  # Numerical features
    X_bin_train = pd.DataFrame(X_train[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_train is not None and np.save(os.path.join(save_path, "X_bin_train.npy"), X_bin_train)  # Binary features
    X_cat_train = pd.DataFrame(X_train[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_train is not None and np.save(os.path.join(save_path, "X_cat_train.npy"), X_cat_train)  # Categorical features
    Y_train = y_train.astype(np.float32) if task_type == "regression" else y_train.astype(np.int64)
    U_value, V_index, U_counts = torch.unique(torch.tensor(Y_train), return_inverse=True, return_counts=True)
    print('U_value', U_value)
    print('U_counts', U_counts)
    np.save(os.path.join(save_path, "Y_train.npy"), Y_train)  # Labels

    ##validation set
    X_num_valid = pd.DataFrame(X_valid[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_valid is not None and np.save(os.path.join(save_path, "X_num_val.npy"), X_num_valid)  # Numerical features
    X_bin_valid = pd.DataFrame(X_valid[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_valid is not None and np.save(os.path.join(save_path, "X_bin_val.npy"), X_bin_valid)  # Binary features
    X_cat_valid = pd.DataFrame(X_valid[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_valid is not None and np.save(os.path.join(save_path, "X_cat_val.npy"), X_cat_valid)  # Categorical features
    # Y_valid = y_valid  ##因为要在后续程序中用到原始class rank
    Y_valid = y_valid.astype(np.float32) if task_type == "regression" else y_valid.astype(np.int64)
    U_value, V_index, U_counts = torch.unique(torch.tensor(Y_valid), return_inverse=True, return_counts=True)
    print('U_value', U_value)
    print('U_counts', U_counts)
    np.save(os.path.join(save_path, "Y_val.npy"), Y_valid)  # Labels

    ##test set
    X_num_test = pd.DataFrame(X_test[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_test is not None and np.save(os.path.join(save_path, "X_num_test.npy"), X_num_test)  # Numerical features
    X_bin_test = pd.DataFrame(X_test[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_test is not None and np.save(os.path.join(save_path, "X_bin_test.npy"), X_bin_test)  # Binary features
    X_cat_test = pd.DataFrame(X_test[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_test is not None and np.save(os.path.join(save_path, "X_cat_test.npy"), X_cat_test)  # Categorical features
    # Y_test = y_test  ##因为要在后续程序中用到原始class rank
    Y_test = y_test.astype(np.float32)if task_type == "regression" else y_test.astype(np.int64)
    U_value, V_index, U_counts = torch.unique(torch.tensor(Y_test), return_inverse=True, return_counts=True)
    print('U_value', U_value)
    print('U_counts', U_counts)
    np.save(os.path.join(save_path, "Y_test.npy"), Y_test)  # Labels

    # info.json
    info = {
        "name": "MSLR-WEB10K(Fold-1)",
        "id": "microsoft--s1200192c5",
        "task_type": task_type,
        "n_num_features": len(numerical_feature_indices),
        "n_bin_features": len(binary_feature_indices),
        "n_cat_features": len(categorical_feature_indices),
        "train_size": X_train.shape[0],
        "val_size": X_valid.shape[0],
        "test_size": X_test.shape[0]
    }
    with open(os.path.join(save_path, "info.json"), "w") as f:
        json.dump(info, f, indent=4)


def preprocess_winequality_regression(data_path, save_path,task_type):
    data = pd.read_csv(data_path, header=None, skiprows=1, sep='\t')
    df_data = data.iloc[:, 0].str.split(';', expand=True)  # 将单列数据按分号分割为多列
    df_data = df_data.map(convert_to_numeric)
    column_types = df_data.dtypes
    sample_count = df_data.shape[0]  # row
    feature_count = df_data.shape[1]  # column

    # 分割数据集
    train_data, temp_data = train_test_split(df_data, train_size=4547, random_state=119)  # wine_quality
    # val_data, test_data  = train_test_split(temp_data, train_size=585, random_state=119)##先划分val_data=585会出现val少分一类，所以先分多的set
    test_data, val_data = train_test_split(temp_data, train_size=1365, random_state=119)  # wine_quality


    X_train, y_train = train_data.iloc[:, 0:11].values, train_data.iloc[:, 11].values
    X_valid, y_valid = val_data.iloc[:, 0:11].values, val_data.iloc[:, 11].values
    X_test, y_test = test_data.iloc[:, 0:11].values, test_data.iloc[:, 11].values
    u_value, u_index, u_counts = torch.unique(torch.tensor(y_train), return_inverse=True, return_counts=True)
    print(u_value)
    print(u_counts)

    df_x = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)], axis=0)  ## Concatenate train, valid, and test dataframes
    df_x = df_x.map(convert_to_numeric) ##将DataFrame中的数据转换为数值型，但保留无法转换的值为字符串类型
    column_types = df_x.dtypes
    numerical_features, binary_features, categorical_features = get_feature_types(df_x)## Get feature types
    # Get indices of feature types
    numerical_feature_indices = [df_x.columns.get_loc(feature) for feature in numerical_features]
    binary_feature_indices = [df_x.columns.get_loc(feature) for feature in binary_features]
    categorical_feature_indices = [df_x.columns.get_loc(feature) for feature in categorical_features]

    ##train set
    X_num_train = pd.DataFrame(X_train[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_train is not None and np.save(os.path.join(save_path, "X_num_train.npy"), X_num_train)  # Numerical features
    X_bin_train = pd.DataFrame(X_train[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_train is not None and np.save(os.path.join(save_path, "X_bin_train.npy"), X_bin_train)  # Binary features
    X_cat_train = pd.DataFrame(X_train[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_train is not None and np.save(os.path.join(save_path, "X_cat_train.npy"), X_cat_train)  # Categorical features
    Y_train = y_train.astype(np.float32) if task_type == "regression" else y_train.astype(np.int64)-3 #np.float32 for regression, np.int64 for classification,the labels must form the range [0, ..., n_classes - 1].
    U_value, V_index, U_counts = torch.unique(torch.tensor(Y_train),return_inverse=True, return_counts=True)
    print('U_value',U_value)
    print('U_counts',U_counts)
    np.save(os.path.join(save_path, "Y_train.npy"), Y_train)  # Labels

    ##validation set
    X_num_valid = pd.DataFrame(X_valid[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_valid is not None and np.save(os.path.join(save_path, "X_num_val.npy"), X_num_valid)  # Numerical features
    X_bin_valid = pd.DataFrame(X_valid[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_valid is not None and np.save(os.path.join(save_path, "X_bin_val.npy"), X_bin_valid)  # Binary features
    X_cat_valid = pd.DataFrame(X_valid[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_valid is not None and np.save(os.path.join(save_path, "X_cat_val.npy"), X_cat_valid)  # Categorical features
    Y_valid = y_valid.astype(np.float32) if task_type == "regression" else y_valid.astype(np.int64)-3
    U_value, V_index, U_counts = torch.unique(torch.tensor(Y_valid), return_inverse=True, return_counts=True)
    print('U_value', U_value)
    print('U_counts', U_counts)
    np.save(os.path.join(save_path, "Y_val.npy"), Y_valid)  # Labels

    ##test set
    X_num_test = pd.DataFrame(X_test[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_test is not None and np.save(os.path.join(save_path, "X_num_test.npy"), X_num_test)  # Numerical features
    X_bin_test = pd.DataFrame(X_test[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_test is not None and np.save(os.path.join(save_path, "X_bin_test.npy"), X_bin_test)  # Binary features
    X_cat_test = pd.DataFrame(X_test[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_test is not None and np.save(os.path.join(save_path, "X_cat_test.npy"), X_cat_test)  # Categorical features
    Y_test = y_test.astype(np.float32) if task_type == "regression" else y_test.astype(np.int64)-3
    U_value, V_index, U_counts = torch.unique(torch.tensor(Y_test), return_inverse=True, return_counts=True)
    print('U_value', U_value)
    print('U_counts', U_counts)
    np.save(os.path.join(save_path, "Y_test.npy"), Y_test)  # Labels


    info = {
        "name": 'winequality',
        "id": "winequality--s6497c7",
        "task_type": task_type,
        "n_num_features": len(numerical_feature_indices),
        "n_bin_features": len(binary_feature_indices),
        "n_cat_features": len(categorical_feature_indices),
        "train_size": X_train.shape[0],
        "val_size": X_valid.shape[0],
        "test_size": X_test.shape[0]
    }
    with open(os.path.join(save_path, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

def preprocess_abalone_regression(input_path,output_path,type_task):
    data = pd.read_csv(input_path, header=None, skiprows=1, sep='\t')
    df_data = data.iloc[:, 0].str.split(',', expand=True) # 将单列数据按','分割为多列
    df_data = df_data.map(convert_to_numeric)
    column_types = df_data.dtypes
    sample_count = df_data.shape[0] #row
    feature_count = df_data.shape[1] #column

    # 分割数据集
    train_data, temp_data = train_test_split(df_data, train_size=int(sample_count*0.7), random_state=119)#abalone
    val_data, test_data = train_test_split(temp_data, train_size=int(sample_count*0.3*(2/3)), random_state=119)#abalone
    X_train, y_train = train_data.iloc[:, 1:8].values, train_data.iloc[:, 0].values
    X_valid, y_valid = val_data.iloc[:, 1:8].values, val_data.iloc[:, 0].values
    X_test, y_test = test_data.iloc[:, 1:8].values, test_data.iloc[:, 0].values
    u_value, u_index, u_counts = torch.unique(torch.tensor(y_train), return_inverse=True, return_counts=True)
    print(u_value)
    print(u_counts)

    df_x = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)], axis=0)
    column_types = df_x.dtypes
    df_x = df_x.map(convert_to_numeric) ##将DataFrame中的数据转换为数值型，但保留无法转换的值为字符串类型
    column_types = df_x.dtypes
    numerical_features, binary_features, categorical_features = get_feature_types(df_x)## Get feature types
    # Get indices of feature types
    numerical_feature_indices = [df_x.columns.get_loc(feature) for feature in numerical_features]
    binary_feature_indices = [df_x.columns.get_loc(feature) for feature in binary_features]
    categorical_feature_indices = [df_x.columns.get_loc(feature) for feature in categorical_features]

    ##train set
    X_num_train = pd.DataFrame(X_train[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_train is not None and np.save(os.path.join(save_path, "X_num_train.npy"), X_num_train)  # Numerical features
    X_bin_train = pd.DataFrame(X_train[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_train is not None and np.save(os.path.join(save_path, "X_bin_train.npy"), X_bin_train)  # Binary features
    X_cat_train = pd.DataFrame(X_train[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_train is not None and np.save(os.path.join(save_path, "X_cat_train.npy"), X_cat_train)  # Categorical features
    # Y_train = y_train  ##因为要在后续程序中用到原始class rank
    Y_train = y_train.astype(np.float32) if task_type == "regression" else y_train.astype(np.int64)-1
    U_value, V_index, U_counts = torch.unique(torch.tensor(Y_train),return_inverse=True, return_counts=True)
    print('U_value',U_value)
    print('U_counts',U_counts)
    np.save(os.path.join(output_path, "Y_train.npy"), Y_train)  # Labels

    ##validation set
    X_num_valid = pd.DataFrame(X_valid[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_valid is not None and np.save(os.path.join(save_path, "X_num_val.npy"), X_num_valid)  # Numerical features
    X_bin_valid = pd.DataFrame(X_valid[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_valid is not None and np.save(os.path.join(save_path, "X_bin_val.npy"), X_bin_valid)  # Binary features
    X_cat_valid = pd.DataFrame(X_valid[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_valid is not None and np.save(os.path.join(save_path, "X_cat_val.npy"), X_cat_valid)  # Categorical features
    # Y_valid = y_valid  ##因为要在后续程序中用到原始class rank
    Y_valid = y_valid.astype(np.float32) if task_type == "regression" else y_valid.astype(np.int64) - 1
    U_value, V_index, U_counts = torch.unique(torch.tensor(Y_valid), return_inverse=True, return_counts=True)
    print('U_value', U_value)
    print('U_counts', U_counts)
    np.save(os.path.join(output_path, "Y_val.npy"), Y_valid)  # Labels

    ##test set
    X_num_test = pd.DataFrame(X_test[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_test is not None and np.save(os.path.join(save_path, "X_num_test.npy"), X_num_test)  # Numerical features
    X_bin_test = pd.DataFrame(X_test[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_test is not None and np.save(os.path.join(save_path, "X_bin_test.npy"), X_bin_test)  # Binary features
    X_cat_test = pd.DataFrame(X_test[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_test is not None and np.save(os.path.join(save_path, "X_cat_test.npy"), X_cat_test)  # Categorical features
    # Y_test = y_test  ##因为要在后续程序中用到原始class rank
    Y_test = y_test.astype(np.float32) if task_type == "regression" else y_test.astype(np.int64) - 1
    U_value, V_index, U_counts = torch.unique(torch.tensor(Y_test), return_inverse=True, return_counts=True)
    print('U_value', U_value)
    print('U_counts', U_counts)
    np.save(os.path.join(output_path, "Y_test.npy"), Y_test)  # Labels

    info = {
        "name": 'abalone',
        "id": "abalone--s4177c8",
        "task_type": type_task,  ##multiclass,regression
        "n_num_features": len(numerical_feature_indices),
        "n_bin_features": len(binary_feature_indices),
        "n_cat_features": len(categorical_feature_indices),
        "train_size": X_train.shape[0],
        "val_size": X_valid.shape[0],
        "test_size": X_test.shape[0]
    }
    with open(os.path.join(save_path, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

def preprocess_eucalyptus_regression(input_path,output_path,type_task):
    data = pd.read_csv(input_path, header=None, skiprows=1, sep='\t')
    df_data = data.iloc[:, 0].str.split(',', expand=True) # 将单列数据按','分割为多列
    df_data = df_data.map(convert_to_numeric)
    column_types = df_data.dtypes
    sample_count = df_data.shape[0] #row
    feature_count = df_data.shape[1] #column

    # 分割数据集
    train_data, temp_data = train_test_split(df_data, train_size=int(sample_count*0.7), random_state=119)#abalone
    val_data, test_data = train_test_split(temp_data, train_size=int(sample_count*0.3*(2/3)), random_state=119)#abalone

    X_train, y_train = train_data.iloc[:,0:19].values, train_data.iloc[:, 20].values
    X_valid, y_valid = val_data.iloc[:,0:19].values, val_data.iloc[:, 20].values
    X_test, y_test = test_data.iloc[:,0:19].values, test_data.iloc[:, 20].values
    u_value, u_index, u_counts = torch.unique(torch.tensor(y_train), return_inverse=True, return_counts=True)
    print(u_value)
    print(u_counts)

    df_x = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)], axis=0,)##合并之后可能导致所有列为object类型
    column_types = df_x.dtypes
    df_x = df_x.map(convert_to_numeric) ##将DataFrame中的数据转换为数值型，但保留无法转换的值为字符串类型
    column_types = df_x.dtypes
    numerical_features, binary_features, categorical_features = get_feature_types(df_x)## Get feature types
    # Get indices of feature types
    numerical_feature_indices = [df_x.columns.get_loc(feature) for feature in numerical_features]
    binary_feature_indices = [df_x.columns.get_loc(feature) for feature in binary_features]
    categorical_feature_indices = [df_x.columns.get_loc(feature) for feature in categorical_features]

    ##train set
    X_num_train = pd.DataFrame(X_train[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_train is not None and np.save(os.path.join(save_path, "X_num_train.npy"), X_num_train)  # Numerical features
    X_bin_train = pd.DataFrame(X_train[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_train is not None and np.save(os.path.join(save_path, "X_bin_train.npy"), X_bin_train)  # Binary features
    X_cat_train = pd.DataFrame(X_train[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_train is not None and np.save(os.path.join(save_path, "X_cat_train.npy"), X_cat_train)  # Categorical features
    # Y_train = y_train  ##因为要在后续程序中用到原始class rank
    Y_train = y_train.astype(np.float32) if task_type == "regression" else y_train.astype(np.int64)-1
    U_value, V_index, U_counts = torch.unique(torch.tensor(Y_train),return_inverse=True, return_counts=True)
    print('U_value',U_value)
    print('U_counts',U_counts)
    np.save(os.path.join(output_path, "Y_train.npy"), Y_train)  # Labels

    ##validation set
    X_num_valid = pd.DataFrame(X_valid[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_valid is not None and np.save(os.path.join(save_path, "X_num_val.npy"), X_num_valid)  # Numerical features
    X_bin_valid = pd.DataFrame(X_valid[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_valid is not None and np.save(os.path.join(save_path, "X_bin_val.npy"), X_bin_valid)  # Binary features
    X_cat_valid = pd.DataFrame(X_valid[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_valid is not None and np.save(os.path.join(save_path, "X_cat_val.npy"), X_cat_valid)  # Categorical features
    # Y_valid = y_valid  ##因为要在后续程序中用到原始class rank
    Y_valid = y_valid.astype(np.float32)if task_type == "regression" else y_valid.astype(np.int64) - 1
    U_value, V_index, U_counts = torch.unique(torch.tensor(Y_valid), return_inverse=True, return_counts=True)
    print('U_value', U_value)
    print('U_counts', U_counts)
    np.save(os.path.join(output_path, "Y_val.npy"), Y_valid)  # Labels

    ##test set
    X_num_test = pd.DataFrame(X_test[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_test is not None and np.save(os.path.join(save_path, "X_num_test.npy"), X_num_test)  # Numerical features
    X_bin_test = pd.DataFrame(X_test[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_test is not None and np.save(os.path.join(save_path, "X_bin_test.npy"), X_bin_test)  # Binary features
    X_cat_test = pd.DataFrame(X_test[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_test is not None and np.save(os.path.join(save_path, "X_cat_test.npy"), X_cat_test)  # Categorical features
    # Y_test = y_test  ##因为要在后续程序中用到原始class rank
    Y_test = y_test.astype(np.float32) if task_type == "regression" else y_test.astype(np.int64) - 1
    U_value, V_index, U_counts = torch.unique(torch.tensor(Y_test), return_inverse=True, return_counts=True)
    print('U_value', U_value)
    print('U_counts', U_counts)
    np.save(os.path.join(output_path, "Y_test.npy"), Y_test)  # Labels

    info = {
        "name": 'eucalyptus',
        "id": "eucalyptus--s736c5",
        "task_type": type_task,  ##multiclass,regression
        "n_num_features": len(numerical_feature_indices),
        "n_bin_features": len(binary_feature_indices),
        "n_cat_features": len(categorical_feature_indices),
        "train_size": X_train.shape[0],
        "val_size": X_valid.shape[0],
        "test_size": X_test.shape[0]
    }
    with open(os.path.join(save_path, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

def preprocess_user_knowledge_regression(input_path,output_path,type_task):
    data, meta = arff.loadarff(input_path)
    df = pd.DataFrame(data)
    df_data = df.map(convert_to_numeric)
    column_types = df_data.dtypes
    sample_count = df_data.shape[0] #row
    feature_count = df_data.shape[1] #column

    # 分割数据集
    train_data, temp_data = train_test_split(df_data, train_size=int(sample_count*0.7), random_state=119)#abalone
    val_data, test_data = train_test_split(temp_data, train_size=int(sample_count*0.3*(2/3)), random_state=119)#abalone

    X_train, y_train = train_data.iloc[:,0:5].values, train_data.iloc[:, 5].values
    X_valid, y_valid = val_data.iloc[:,0:5].values, val_data.iloc[:, 5].values
    X_test, y_test = test_data.iloc[:,0:5].values, test_data.iloc[:, 5].values
    u_value, u_index, u_counts = torch.unique(torch.tensor(y_train), return_inverse=True, return_counts=True)
    print(u_value)
    print(u_counts)

    df_x = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)], axis=0,)##合并之后可能导致所有列为object类型
    column_types = df_x.dtypes
    df_x = df_x.map(convert_to_numeric) ##将DataFrame中的数据转换为数值型，但保留无法转换的值为字符串类型
    column_types = df_x.dtypes
    numerical_features, binary_features, categorical_features = get_feature_types(df_x)## Get feature types
    # Get indices of feature types
    numerical_feature_indices = [df_x.columns.get_loc(feature) for feature in numerical_features]
    binary_feature_indices = [df_x.columns.get_loc(feature) for feature in binary_features]
    categorical_feature_indices = [df_x.columns.get_loc(feature) for feature in categorical_features]

    ##train set
    X_num_train = pd.DataFrame(X_train[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_train is not None and np.save(os.path.join(save_path, "X_num_train.npy"), X_num_train)  # Numerical features
    X_bin_train = pd.DataFrame(X_train[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_train is not None and np.save(os.path.join(save_path, "X_bin_train.npy"), X_bin_train)  # Binary features
    X_cat_train = pd.DataFrame(X_train[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_train is not None and np.save(os.path.join(save_path, "X_cat_train.npy"), X_cat_train)  # Categorical features
    Y_train = y_train.astype(float)  ##因为要在后续程序中用到原始class rank
    np.save(os.path.join(output_path, "Y_train.npy"), Y_train)  # Labels

    ##validation set
    X_num_valid = pd.DataFrame(X_valid[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_valid is not None and np.save(os.path.join(save_path, "X_num_val.npy"), X_num_valid)  # Numerical features
    X_bin_valid = pd.DataFrame(X_valid[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_valid is not None and np.save(os.path.join(save_path, "X_bin_val.npy"), X_bin_valid)  # Binary features
    X_cat_valid = pd.DataFrame(X_valid[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_valid is not None and np.save(os.path.join(save_path, "X_cat_val.npy"), X_cat_valid)  # Categorical features
    Y_valid = y_valid.astype(float)  ##因为要在后续程序中用到原始class rank
    np.save(os.path.join(output_path, "Y_val.npy"), Y_valid)  # Labels

    ##test set
    X_num_test = pd.DataFrame(X_test[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_test is not None and np.save(os.path.join(save_path, "X_num_test.npy"), X_num_test)  # Numerical features
    X_bin_test = pd.DataFrame(X_test[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_test is not None and np.save(os.path.join(save_path, "X_bin_test.npy"), X_bin_test)  # Binary features
    X_cat_test = pd.DataFrame(X_test[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_test is not None and np.save(os.path.join(save_path, "X_cat_test.npy"), X_cat_test)  # Categorical features
    Y_test = y_test.astype(float)  ##因为要在后续程序中用到原始class rank
    np.save(os.path.join(output_path, "Y_test.npy"), Y_test)  # Labels

    info = {
        "name": 'user-knowledge',
        "id": "user-knowledge--s403c5",
        "task_type": type_task,  ##multiclass,regression
        "n_num_features": len(numerical_feature_indices),
        "n_bin_features": len(binary_feature_indices),
        "n_cat_features": len(categorical_feature_indices),
        "train_size": X_train.shape[0],
        "val_size": X_valid.shape[0],
        "test_size": X_test.shape[0]
    }
    with open(os.path.join(save_path, "info.json"), "w") as f:
        json.dump(info, f, indent=4)
def preprocess_eye_movements_regression(input_path,output_path,type_task):
    data, meta = arff.loadarff(input_path)
    df = pd.DataFrame(data)
    df_data = df.map(convert_to_numeric)
    column_types = df_data.dtypes
    sample_count = df_data.shape[0] #row
    feature_count = df_data.shape[1] #column

    # 分割数据集
    train_data, temp_data = train_test_split(df_data, train_size=int(sample_count*0.7), random_state=119)#abalone
    val_data, test_data = train_test_split(temp_data, train_size=int(sample_count*0.3*(2/3)), random_state=119)#abalone

    X_train, y_train = train_data.iloc[:,1:27].values, train_data.iloc[:, 27].values
    X_valid, y_valid = val_data.iloc[:,1:27].values, val_data.iloc[:, 27].values
    X_test, y_test = test_data.iloc[:,1:27].values, test_data.iloc[:, 27].values
    u_value, u_index, u_counts = torch.unique(torch.tensor(y_train), return_inverse=True, return_counts=True)
    print(u_value)
    print(u_counts)

    df_x = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)], axis=0,)##合并之后可能导致所有列为object类型
    column_types = df_x.dtypes
    df_x = df_x.map(convert_to_numeric) ##将DataFrame中的数据转换为数值型，但保留无法转换的值为字符串类型
    column_types = df_x.dtypes
    numerical_features, binary_features, categorical_features = get_feature_types(df_x)## Get feature types
    # Get indices of feature types
    numerical_feature_indices = [df_x.columns.get_loc(feature) for feature in numerical_features]
    binary_feature_indices = [df_x.columns.get_loc(feature) for feature in binary_features]
    categorical_feature_indices = [df_x.columns.get_loc(feature) for feature in categorical_features]

    ##train set
    X_num_train = pd.DataFrame(X_train[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_train is not None and np.save(os.path.join(save_path, "X_num_train.npy"), X_num_train)  # Numerical features
    X_bin_train = pd.DataFrame(X_train[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_train is not None and np.save(os.path.join(save_path, "X_bin_train.npy"), X_bin_train)  # Binary features
    X_cat_train = pd.DataFrame(X_train[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_train is not None and np.save(os.path.join(save_path, "X_cat_train.npy"), X_cat_train)  # Categorical features
    Y_train = y_train.astype(float)  ##因为要在后续程序中用到原始class rank
    np.save(os.path.join(output_path, "Y_train.npy"), Y_train)  # Labels

    ##validation set
    X_num_valid = pd.DataFrame(X_valid[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_valid is not None and np.save(os.path.join(save_path, "X_num_val.npy"), X_num_valid)  # Numerical features
    X_bin_valid = pd.DataFrame(X_valid[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_valid is not None and np.save(os.path.join(save_path, "X_bin_val.npy"), X_bin_valid)  # Binary features
    X_cat_valid = pd.DataFrame(X_valid[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_valid is not None and np.save(os.path.join(save_path, "X_cat_val.npy"), X_cat_valid)  # Categorical features
    Y_valid = y_valid.astype(float)  ##因为要在后续程序中用到原始class rank
    np.save(os.path.join(output_path, "Y_val.npy"), Y_valid)  # Labels

    ##test set
    X_num_test = pd.DataFrame(X_test[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_test is not None and np.save(os.path.join(save_path, "X_num_test.npy"), X_num_test)  # Numerical features
    X_bin_test = pd.DataFrame(X_test[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_test is not None and np.save(os.path.join(save_path, "X_bin_test.npy"), X_bin_test)  # Binary features
    X_cat_test = pd.DataFrame(X_test[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_test is not None and np.save(os.path.join(save_path, "X_cat_test.npy"), X_cat_test)  # Categorical features
    Y_test = y_test.astype(float)  ##因为要在后续程序中用到原始class rank
    np.save(os.path.join(output_path, "Y_test.npy"), Y_test)  # Labels

    info = {
        "name": 'eye_movements',
        "id": "eye_movements--s10936c3",
        "task_type": type_task,  ##multiclass,regression
        "n_num_features": len(numerical_feature_indices),
        "n_bin_features": len(binary_feature_indices),
        "n_cat_features": len(categorical_feature_indices),
        "train_size": X_train.shape[0],
        "val_size": X_valid.shape[0],
        "test_size": X_test.shape[0]
    }
    with open(os.path.join(save_path, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

def preprocess_autos_regression(input_path,output_path,type_task):
    data, meta = arff.loadarff(input_path)
    df = pd.DataFrame(data)
    df.fillna(-2024, inplace=True)##缺失值赋值为-2024
    df.replace('?', -2024, inplace=True)

    object_columns = df.select_dtypes(include=['object']).columns
    df[object_columns] = df[object_columns].astype(str).astype('string')
    # for column in object_columns:
    #     values = df[column].values
    #     for i in range(len(values)):
    #         values[i] = str(values[i])
    #     df[column]=values
    # df[object_columns] = df[object_columns].astype('string')
    df_data = df.map(convert_to_numeric)
    column_types = df_data.dtypes
    sample_count = df_data.shape[0] #row
    feature_count = df_data.shape[1] #column

    # 分割数据集
    df_data.iloc[:, -1] = df_data.iloc[:, -1].replace({-3:0, -2:1, -1:2, 0:3, 1:4, 2:5, 3:6})#最后一列标签从-3，-2，-1，0，1，2，3变为0，1，2，3，4，5，6
    train_data, temp_data = train_test_split(df_data, train_size=int(sample_count*0.7), random_state=119)#abalone
    val_data, test_data = train_test_split(temp_data, train_size=int(sample_count*0.3*(2/3)), random_state=119)#abalone

    X_train, y_train = train_data.iloc[:,0:25].values, train_data.iloc[:, 25].values
    X_train[X_train == '?'] = 'wenhao'
    X_valid, y_valid = val_data.iloc[:,0:25].values, val_data.iloc[:, 25].values
    X_valid[X_valid == '?'] = 'wenhao'
    X_test, y_test = test_data.iloc[:,0:25].values, test_data.iloc[:, 25].values
    X_test[X_test == '?'] = 'wenhao'
    u_value, u_index, u_counts = torch.unique(torch.tensor(y_train), return_inverse=True, return_counts=True)
    print(u_value)
    print(u_counts)

    df_x = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)], axis=0,)##合并之后可能导致所有列为object类型
    column_types = df_x.dtypes
    df_x = df_x.map(convert_to_numeric) ##将DataFrame中的数据转换为数值型，但保留无法转换的值为字符串类型
    column_types = df_x.dtypes

    numerical_features, binary_features, categorical_features = get_feature_types(df_x)## Get feature types
    # Get indices of feature types
    numerical_feature_indices = [df_x.columns.get_loc(feature) for feature in numerical_features]
    binary_feature_indices = [df_x.columns.get_loc(feature) for feature in binary_features]
    categorical_feature_indices = [df_x.columns.get_loc(feature) for feature in categorical_features]

    ##train set
    X_num_train = X_train[:, numerical_feature_indices].astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_train is not None and np.save(os.path.join(save_path, "X_num_train.npy"), X_num_train)  # Numerical features
    X_bin_train = X_train[:, binary_feature_indices].astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_train is not None and np.save(os.path.join(save_path, "X_bin_train.npy"), X_bin_train)  # Binary features
    X_cat_train =X_train[:, categorical_feature_indices].astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_train is not None and np.save(os.path.join(save_path, "X_cat_train.npy"), X_cat_train)  # Categorical features
    Y_train = y_train.astype(float)  ##因为要在后续程序中用到原始class rank
    np.save(os.path.join(output_path, "Y_train.npy"), Y_train)  # Labels

    ##validation set
    X_num_valid = X_valid[:, numerical_feature_indices].astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_valid is not None and np.save(os.path.join(save_path, "X_num_val.npy"), X_num_valid)  # Numerical features
    X_bin_valid = X_valid[:, binary_feature_indices].astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_valid is not None and np.save(os.path.join(save_path, "X_bin_val.npy"), X_bin_valid)  # Binary features
    X_cat_valid = X_valid[:, categorical_feature_indices].astype(str)if len(categorical_feature_indices) > 0 else None

    X_cat_valid is not None and np.save(os.path.join(save_path, "X_cat_val.npy"), X_cat_valid)  # Categorical features
    Y_valid = y_valid.astype(float)  ##因为要在后续程序中用到原始class rank
    np.save(os.path.join(output_path, "Y_val.npy"), Y_valid)  # Labels

    ##test set
    X_num_test = X_test[:, numerical_feature_indices].astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_test is not None and np.save(os.path.join(save_path, "X_num_test.npy"), X_num_test)  # Numerical features
    X_bin_test = X_test[:, binary_feature_indices].astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_test is not None and np.save(os.path.join(save_path, "X_bin_test.npy"), X_bin_test)  # Binary features
    X_cat_test = X_test[:, categorical_feature_indices].astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_test is not None and np.save(os.path.join(save_path, "X_cat_test.npy"), X_cat_test)  # Categorical features
    Y_test = y_test.astype(float)  ##因为要在后续程序中用到原始class rank
    np.save(os.path.join(output_path, "Y_test.npy"), Y_test)  # Labels

    info = {
        "name": 'autos',
        "id": "autos--s205c7",
        "task_type": type_task,  ##multiclass,regression
        "n_num_features": len(numerical_feature_indices),
        "n_bin_features": len(binary_feature_indices),
        "n_cat_features": len(categorical_feature_indices),
        "train_size": X_train.shape[0],
        "val_size": X_valid.shape[0],
        "test_size": X_test.shape[0]
    }
    with open(os.path.join(save_path, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

def preprocess_cmc_regression(input_path,output_path,type_task):
    data, meta = arff.loadarff(input_path)
    df = pd.DataFrame(data)
    df_data = df.map(convert_to_numeric)
    column_types = df_data.dtypes
    sample_count = df_data.shape[0] #row
    feature_count = df_data.shape[1] #column

    # 分割数据集
    df_data.iloc[:, -1] = df_data.iloc[:, -1].replace({3: 2, 2: 3}) # 将标签列的2和3互换，因为原数据集2代表长期避孕，3代表短期，0代表无
    train_data, temp_data = train_test_split(df_data, train_size=int(sample_count*0.7), random_state=119)#abalone
    val_data, test_data = train_test_split(temp_data, train_size=int(sample_count*0.3*(2/3)), random_state=119)#abalone

    X_train, y_train = train_data.iloc[:,0:9].values, train_data.iloc[:, 9].values
    X_valid, y_valid = val_data.iloc[:,0:9].values, val_data.iloc[:, 9].values
    X_test, y_test = test_data.iloc[:,0:9].values, test_data.iloc[:, 9].values
    u_value, u_index, u_counts = torch.unique(torch.tensor(y_train), return_inverse=True, return_counts=True)
    print(u_value)
    print(u_counts)

    df_x = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)], axis=0,)##合并之后可能导致所有列为object类型
    column_types = df_x.dtypes
    df_x = df_x.map(convert_to_numeric) ##将DataFrame中的数据转换为数值型，但保留无法转换的值为字符串类型
    column_types = df_x.dtypes
    numerical_features, binary_features, categorical_features = get_feature_types(df_x)## Get feature types
    # Get indices of feature types
    numerical_feature_indices = [df_x.columns.get_loc(feature) for feature in numerical_features]
    binary_feature_indices = [df_x.columns.get_loc(feature) for feature in binary_features]
    categorical_feature_indices = [df_x.columns.get_loc(feature) for feature in categorical_features]

    ##train set
    X_num_train = pd.DataFrame(X_train[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_train is not None and np.save(os.path.join(save_path, "X_num_train.npy"), X_num_train)  # Numerical features
    X_bin_train = pd.DataFrame(X_train[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_train is not None and np.save(os.path.join(save_path, "X_bin_train.npy"), X_bin_train)  # Binary features
    X_cat_train = pd.DataFrame(X_train[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_train is not None and np.save(os.path.join(save_path, "X_cat_train.npy"), X_cat_train)  # Categorical features
    # Y_train = y_train  ##因为要在后续程序中用到原始class rank
    Y_train = y_train.astype(np.float32) if task_type == "regression" else y_train.astype(np.int64)-1
    U_value, V_index, U_counts = torch.unique(torch.tensor(Y_train),return_inverse=True, return_counts=True)
    print('U_value',U_value)
    print('U_counts',U_counts)
    np.save(os.path.join(output_path, "Y_train.npy"), Y_train)  # Labels

    ##validation set
    X_num_valid = pd.DataFrame(X_valid[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_valid is not None and np.save(os.path.join(save_path, "X_num_val.npy"), X_num_valid)  # Numerical features
    X_bin_valid = pd.DataFrame(X_valid[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_valid is not None and np.save(os.path.join(save_path, "X_bin_val.npy"), X_bin_valid)  # Binary features
    X_cat_valid = pd.DataFrame(X_valid[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_valid is not None and np.save(os.path.join(save_path, "X_cat_val.npy"), X_cat_valid)  # Categorical features
    # Y_valid = y_valid  ##因为要在后续程序中用到原始class rank
    Y_valid = y_valid.astype(np.float32) if task_type == "regression" else y_valid.astype(np.int64) - 1
    U_value, V_index, U_counts = torch.unique(torch.tensor(Y_valid), return_inverse=True, return_counts=True)
    print('U_value', U_value)
    print('U_counts', U_counts)
    np.save(os.path.join(output_path, "Y_val.npy"), Y_valid)  # Labels

    ##test set
    X_num_test = pd.DataFrame(X_test[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_test is not None and np.save(os.path.join(save_path, "X_num_test.npy"), X_num_test)  # Numerical features
    X_bin_test = pd.DataFrame(X_test[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_test is not None and np.save(os.path.join(save_path, "X_bin_test.npy"), X_bin_test)  # Binary features
    X_cat_test = pd.DataFrame(X_test[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_test is not None and np.save(os.path.join(save_path, "X_cat_test.npy"), X_cat_test)  # Categorical features
    # Y_test = y_test  ##因为要在后续程序中用到原始class rank
    Y_test = y_test.astype(np.float32) if task_type == "regression" else y_test.astype(np.int64) - 1
    U_value, V_index, U_counts = torch.unique(torch.tensor(Y_test), return_inverse=True, return_counts=True)
    print('U_value', U_value)
    print('U_counts', U_counts)
    np.save(os.path.join(output_path, "Y_test.npy"), Y_test)  # Labels

    info = {
        "name": 'cmc',
        "id": "cmc--s1473c3",
        "task_type": type_task,  ##multiclass,regression
        "n_num_features": len(numerical_feature_indices),
        "n_bin_features": len(binary_feature_indices),
        "n_cat_features": len(categorical_feature_indices),
        "train_size": X_train.shape[0],
        "val_size": X_valid.shape[0],
        "test_size": X_test.shape[0]
    }
    with open(os.path.join(save_path, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

def preprocess_tae_regression(input_path,output_path,type_task):
    data, meta = arff.loadarff(input_path)
    df = pd.DataFrame(data)
    df_data = df.map(convert_to_numeric)
    column_types = df_data.dtypes
    sample_count = df_data.shape[0] #row
    feature_count = df_data.shape[1] #column

    # 分割数据集
    train_data, temp_data = train_test_split(df_data, train_size=int(sample_count*0.7), random_state=119)#abalone
    val_data, test_data = train_test_split(temp_data, train_size=int(sample_count*0.3*(2/3)), random_state=119)#abalone

    X_train, y_train = train_data.iloc[:,0:5].values, train_data.iloc[:, 5].values
    X_valid, y_valid = val_data.iloc[:,0:5].values, val_data.iloc[:, 5].values
    X_test, y_test = test_data.iloc[:,0:5].values, test_data.iloc[:, 5].values
    u_value, u_index, u_counts = torch.unique(torch.tensor(y_train), return_inverse=True, return_counts=True)
    print(u_value)
    print(u_counts)

    df_x = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)], axis=0,)##合并之后可能导致所有列为object类型
    column_types = df_x.dtypes
    df_x = df_x.map(convert_to_numeric) ##将DataFrame中的数据转换为数值型，但保留无法转换的值为字符串类型
    column_types = df_x.dtypes
    numerical_features, binary_features, categorical_features = get_feature_types(df_x)## Get feature types
    # Get indices of feature types
    numerical_feature_indices = [df_x.columns.get_loc(feature) for feature in numerical_features]
    binary_feature_indices = [df_x.columns.get_loc(feature) for feature in binary_features]
    categorical_feature_indices = [df_x.columns.get_loc(feature) for feature in categorical_features]

    ##train set
    X_num_train = pd.DataFrame(X_train[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_train is not None and np.save(os.path.join(save_path, "X_num_train.npy"), X_num_train)  # Numerical features
    X_bin_train = pd.DataFrame(X_train[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_train is not None and np.save(os.path.join(save_path, "X_bin_train.npy"), X_bin_train)  # Binary features
    X_cat_train = pd.DataFrame(X_train[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_train is not None and np.save(os.path.join(save_path, "X_cat_train.npy"), X_cat_train)  # Categorical features
    Y_train = y_train.astype(float)  ##因为要在后续程序中用到原始class rank
    np.save(os.path.join(output_path, "Y_train.npy"), Y_train)  # Labels

    ##validation set
    X_num_valid = pd.DataFrame(X_valid[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_valid is not None and np.save(os.path.join(save_path, "X_num_val.npy"), X_num_valid)  # Numerical features
    X_bin_valid = pd.DataFrame(X_valid[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_valid is not None and np.save(os.path.join(save_path, "X_bin_val.npy"), X_bin_valid)  # Binary features
    X_cat_valid = pd.DataFrame(X_valid[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_valid is not None and np.save(os.path.join(save_path, "X_cat_val.npy"), X_cat_valid)  # Categorical features
    Y_valid = y_valid.astype(float)  ##因为要在后续程序中用到原始class rank
    np.save(os.path.join(output_path, "Y_val.npy"), Y_valid)  # Labels

    ##test set
    X_num_test = pd.DataFrame(X_test[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_test is not None and np.save(os.path.join(save_path, "X_num_test.npy"), X_num_test)  # Numerical features
    X_bin_test = pd.DataFrame(X_test[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_test is not None and np.save(os.path.join(save_path, "X_bin_test.npy"), X_bin_test)  # Binary features
    X_cat_test = pd.DataFrame(X_test[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_test is not None and np.save(os.path.join(save_path, "X_cat_test.npy"), X_cat_test)  # Categorical features
    Y_test = y_test.astype(float)  ##因为要在后续程序中用到原始class rank
    np.save(os.path.join(output_path, "Y_test.npy"), Y_test)  # Labels

    info = {
        "name": 'tae',
        "id": "tae--s151c3",
        "task_type": type_task,  ##multiclass,regression
        "n_num_features": len(numerical_feature_indices),
        "n_bin_features": len(binary_feature_indices),
        "n_cat_features": len(categorical_feature_indices),
        "train_size": X_train.shape[0],
        "val_size": X_valid.shape[0],
        "test_size": X_test.shape[0]
    }
    with open(os.path.join(save_path, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

def preprocess_grub_damage_regression(input_path,output_path,type_task):
    data, meta = arff.loadarff(input_path)
    df = pd.DataFrame(data)
    df_data = df.map(convert_to_numeric)
    column_types = df_data.dtypes
    sample_count = df_data.shape[0] #row
    feature_count = df_data.shape[1] #column

    # 分割数据集
    df_data.iloc[:, -1] = df_data.iloc[:, -1].replace({b'low': 1, b'average': 2, b'high': 3, b'veryhigh': 4}) # {low,average,high,veryhigh}={1,2,3,4}
    train_data, temp_data = train_test_split(df_data, train_size=int(sample_count*0.7), random_state=119)#abalone
    val_data, test_data = train_test_split(temp_data, train_size=int(sample_count*0.3*(2/3)), random_state=119)#abalone

    X_train, y_train = train_data.iloc[:,0:8].values, train_data.iloc[:, 8].values
    X_valid, y_valid = val_data.iloc[:,0:8].values, val_data.iloc[:, 8].values
    X_test, y_test = test_data.iloc[:,0:8].values, test_data.iloc[:, 8].values
    u_value, u_index, u_counts = torch.unique(torch.tensor(y_train.astype(float)), return_inverse=True, return_counts=True)
    print(u_value)
    print(u_counts)

    df_x = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)], axis=0,)##合并之后可能导致所有列为object类型
    column_types = df_x.dtypes
    df_x = df_x.map(convert_to_numeric) ##将DataFrame中的数据转换为数值型，但保留无法转换的值为字符串类型
    column_types = df_x.dtypes
    numerical_features, binary_features, categorical_features = get_feature_types(df_x)## Get feature types
    # Get indices of feature types
    numerical_feature_indices = [df_x.columns.get_loc(feature) for feature in numerical_features]
    binary_feature_indices = [df_x.columns.get_loc(feature) for feature in binary_features]
    categorical_feature_indices = [df_x.columns.get_loc(feature) for feature in categorical_features]

    ##train set
    X_num_train = pd.DataFrame(X_train[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_train is not None and np.save(os.path.join(save_path, "X_num_train.npy"), X_num_train)  # Numerical features
    X_bin_train = pd.DataFrame(X_train[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_train is not None and np.save(os.path.join(save_path, "X_bin_train.npy"), X_bin_train)  # Binary features
    X_cat_train = pd.DataFrame(X_train[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_train is not None and np.save(os.path.join(save_path, "X_cat_train.npy"), X_cat_train)  # Categorical features
    # Y_train = y_train  ##因为要在后续程序中用到原始class rank
    Y_train = y_train.astype(np.float32) if task_type == "regression" else y_train.astype(np.int64) - 1
    U_value, V_index, U_counts = torch.unique(torch.tensor(Y_train), return_inverse=True, return_counts=True)
    print('U_value', U_value)
    print('U_counts', U_counts)
    np.save(os.path.join(output_path, "Y_train.npy"), Y_train)  # Labels

    ##validation set
    X_num_valid = pd.DataFrame(X_valid[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_valid is not None and np.save(os.path.join(save_path, "X_num_val.npy"), X_num_valid)  # Numerical features
    X_bin_valid = pd.DataFrame(X_valid[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_valid is not None and np.save(os.path.join(save_path, "X_bin_val.npy"), X_bin_valid)  # Binary features
    X_cat_valid = pd.DataFrame(X_valid[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_valid is not None and np.save(os.path.join(save_path, "X_cat_val.npy"), X_cat_valid)  # Categorical features
    # Y_valid = y_valid  ##因为要在后续程序中用到原始class rank
    Y_valid = y_valid.astype(np.float32) if task_type == "regression" else y_valid.astype(np.int64) - 1
    U_value, V_index, U_counts = torch.unique(torch.tensor(Y_valid), return_inverse=True, return_counts=True)
    print('U_value', U_value)
    print('U_counts', U_counts)
    np.save(os.path.join(output_path, "Y_val.npy"), Y_valid)  # Labels

    ##test set
    X_num_test = pd.DataFrame(X_test[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_test is not None and np.save(os.path.join(save_path, "X_num_test.npy"), X_num_test)  # Numerical features
    X_bin_test = pd.DataFrame(X_test[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_test is not None and np.save(os.path.join(save_path, "X_bin_test.npy"), X_bin_test)  # Binary features
    X_cat_test = pd.DataFrame(X_test[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_test is not None and np.save(os.path.join(save_path, "X_cat_test.npy"), X_cat_test)  # Categorical features
    # Y_test = y_test  ##因为要在后续程序中用到原始class rank
    Y_test = y_test.astype(np.float32) if task_type == "regression" else y_test.astype(np.int64) - 1
    U_value, V_index, U_counts = torch.unique(torch.tensor(Y_test), return_inverse=True, return_counts=True)
    print('U_value', U_value)
    print('U_counts', U_counts)
    np.save(os.path.join(output_path, "Y_test.npy"), Y_test)  # Labels

    info = {
        "name": 'grub_damage',
        "id": "grub_damage--s155c4",
        "task_type": type_task,  ##multiclass,regression
        "n_num_features": len(numerical_feature_indices),
        "n_bin_features": len(binary_feature_indices),
        "n_cat_features": len(categorical_feature_indices),
        "train_size": X_train.shape[0],
        "val_size": X_valid.shape[0],
        "test_size": X_test.shape[0]
    }
    with open(os.path.join(save_path, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

def preprocess_car_regression(input_path,output_path,type_task):
    data, meta = arff.loadarff(input_path)
    df = pd.DataFrame(data)
    df_data = df.map(convert_to_numeric)
    column_types = df_data.dtypes
    sample_count = df_data.shape[0] #row
    feature_count = df_data.shape[1] #column

    # 分割数据集
    df_data.iloc[:, -1] = df_data.iloc[:, -1].replace({b'unacc': 0, b'acc': 1, b'good': 2, b'vgood': 3})  # {unacc,acc,good,vgood}={0,1,2,3}
    df_data.iloc[:, -5] = df_data.iloc[:, -5].replace({b'5more': 5})# ryx 数据集缺少numerical_features，将5more变成5，增加一列numerical_features

    train_data, temp_data = train_test_split(df_data, train_size=int(sample_count*0.7), random_state=119)#abalone
    val_data, test_data = train_test_split(temp_data, train_size=int(sample_count*0.3*(2/3)), random_state=119)#abalone
    X_train, y_train = train_data.iloc[:,0:6].values, train_data.iloc[:, 6].values
    X_valid, y_valid = val_data.iloc[:,0:6].values, val_data.iloc[:, 6].values
    X_test, y_test = test_data.iloc[:,0:6].values, test_data.iloc[:, 6].values
    u_value, u_index, u_counts = torch.unique(torch.tensor(y_train.astype(int)), return_inverse=True, return_counts=True)
    print(u_value)
    print(u_counts)

    df_x = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)], axis=0,)##合并之后可能导致所有列为object类型
    column_types = df_x.dtypes
    df_x = df_x.map(convert_to_numeric) ##将DataFrame中的数据转换为数值型，但保留无法转换的值为字符串类型
    column_types = df_x.dtypes
    numerical_features, binary_features, categorical_features = get_feature_types(df_x)## Get feature types
    # Get indices of feature types
    numerical_feature_indices = [df_x.columns.get_loc(feature) for feature in numerical_features]
    binary_feature_indices = [df_x.columns.get_loc(feature) for feature in binary_features]
    categorical_feature_indices = [df_x.columns.get_loc(feature) for feature in categorical_features]

    ##train set
    X_num_train = pd.DataFrame(X_train[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_train is not None and np.save(os.path.join(save_path, "X_num_train.npy"), X_num_train)  # Numerical features
    X_bin_train = pd.DataFrame(X_train[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_train is not None and np.save(os.path.join(save_path, "X_bin_train.npy"), X_bin_train)  # Binary features
    X_cat_train = pd.DataFrame(X_train[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_train is not None and np.save(os.path.join(save_path, "X_cat_train.npy"), X_cat_train)  # Categorical features
    # Y_train = y_train  ##因为要在后续程序中用到原始class rank
    Y_train = y_train.astype(np.float32) if task_type == "regression" else y_train.astype(np.int64)
    U_value, V_index, U_counts = torch.unique(torch.tensor(Y_train), return_inverse=True, return_counts=True)
    print('U_value', U_value)
    print('U_counts', U_counts)
    np.save(os.path.join(output_path, "Y_train.npy"), Y_train)  # Labels

    ##validation set
    X_num_valid = pd.DataFrame(X_valid[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_valid is not None and np.save(os.path.join(save_path, "X_num_val.npy"), X_num_valid)  # Numerical features
    X_bin_valid = pd.DataFrame(X_valid[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_valid is not None and np.save(os.path.join(save_path, "X_bin_val.npy"), X_bin_valid)  # Binary features
    X_cat_valid = pd.DataFrame(X_valid[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_valid is not None and np.save(os.path.join(save_path, "X_cat_val.npy"), X_cat_valid)  # Categorical features
    # Y_valid = y_valid  ##因为要在后续程序中用到原始class rank
    Y_valid = y_valid.astype(np.float32) if task_type == "regression" else y_valid.astype(np.int64)
    U_value, V_index, U_counts = torch.unique(torch.tensor(Y_valid), return_inverse=True, return_counts=True)
    print('U_value', U_value)
    print('U_counts', U_counts)
    np.save(os.path.join(output_path, "Y_val.npy"), Y_valid)  # Labels

    ##test set
    X_num_test = pd.DataFrame(X_test[:, numerical_feature_indices]).values.astype(np.float32) if len(numerical_feature_indices) > 0 else None
    X_num_test is not None and np.save(os.path.join(save_path, "X_num_test.npy"), X_num_test)  # Numerical features
    X_bin_test = pd.DataFrame(X_test[:, binary_feature_indices]).values.astype(np.float32) if len(binary_feature_indices) > 0 else None
    X_bin_test is not None and np.save(os.path.join(save_path, "X_bin_test.npy"), X_bin_test)  # Binary features
    X_cat_test = pd.DataFrame(X_test[:, categorical_feature_indices]).values.astype(np.str_) if len(categorical_feature_indices) > 0 else None
    X_cat_test is not None and np.save(os.path.join(save_path, "X_cat_test.npy"), X_cat_test)  # Categorical features
    # Y_test = y_test  ##因为要在后续程序中用到原始class rank
    Y_test = y_test.astype(np.float32) if task_type == "regression" else y_test.astype(np.int64)
    U_value, V_index, U_counts = torch.unique(torch.tensor(Y_test), return_inverse=True, return_counts=True)
    print('U_value', U_value)
    print('U_counts', U_counts)
    np.save(os.path.join(output_path, "Y_test.npy"), Y_test)  # Labels

    info = {
        "name": 'car',
        "id": "car--s1728c4",
        "task_type": type_task,  ##multiclass,regression
        "n_num_features": len(numerical_feature_indices),
        "n_bin_features": len(binary_feature_indices),
        "n_cat_features": len(categorical_feature_indices),
        "train_size": X_train.shape[0],
        "val_size": X_valid.shape[0],
        "test_size": X_test.shape[0]
    }
    with open(os.path.join(save_path, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

if __name__ == "__main__":

# output:yahoo dateset:
#                     # https://www.dropbox.com/s/7rq3ki5vtxm6gzx/yahoo_set_1_train.gz?dl=1
#                     # https://www.dropbox.com/s/3ai8rxm1v0l5sd1/yahoo_set_1_validation.gz?dl=1
#                     # https://www.dropbox.com/s/3d7tdfb1an0b6i4/yahoo_set_1_test.gz?dl=1
#     save_path = '/home/luozhengdong/ordinal_regression/tabR_lzd/data/yahoo_regression/' ##yahoo_classification,yahoo_regression
#     os.makedirs(save_path, exist_ok=True)
#     # input:
#     original_train_path = '/home/luozhengdong/datasets/yahoo/yt___home_mltools_data_pools_ranking_yahoo_set_1_train'
#     new_train_path = '/home/luozhengdong/datasets/yahoo/yt___home_mltools_data_pools_ranking_yahoo_set_1_train.tsv'
#     if os.path.exists(original_train_path):
#         os.rename(original_train_path, new_train_path)
#     original_valid_path = '/home/luozhengdong/datasets/yahoo/yt___home_mltools_data_pools_ranking_yahoo_set_1_validation'
#     new_valid_path = '/home/luozhengdong/datasets/yahoo/yt___home_mltools_data_pools_ranking_yahoo_set_1_validation.tsv'
#     if os.path.exists(original_valid_path):
#         os.rename(original_train_path, original_valid_path)
#     original_test_path = '/home/luozhengdong/datasets/yahoo/yt___home_mltools_data_pools_ranking_yahoo_set_1_test'
#     new_test_path = '/home/luozhengdong/datasets/yahoo/yt___home_mltools_data_pools_ranking_yahoo_set_1_test.tsv'
#     if os.path.exists(original_test_path):
#         os.rename(original_test_path, new_test_path)
#
#     task_type='regression' #regression,multiclass
#     preprocess_yahoo_regression(new_train_path, new_valid_path, new_test_path,task_type)

# output:microsoft dataset Microsoft WEB-10K fold1
    save_path = '/home/luozhengdong/ordinal_regression/tabR_lzd/data/microsoft_classification/'#microsoft_classification
    os.makedirs(save_path, exist_ok=True)
    # input:
    train_path = '/home/luozhengdong/datasets/MICROSOFT/Fold1/train.txt'
    valid_path = '/home/luozhengdong/datasets/MICROSOFT/Fold1/vali.txt'
    test_path = '/home/luozhengdong/datasets/MICROSOFT/Fold1/test.txt'
    task_type = 'multiclass'  ##regression,multiclass
    preprocess_microsoft_regression(train_path, valid_path, test_path, task_type)

# # output:winequality dataset winequality(white+red) #4547,585,1365
#     save_path = '/home/luozhengdong/ordinal_regression/tabR_lzd/data/wine_quality_classification'
#     # save_path = '/home/luozhengdong/ordinal_regression/tabR_lzd/data/regression-wine_quality-red'
#     os.makedirs(save_path, exist_ok=True)
#     # input:
#     data_path = '/home/luozhengdong/datasets/wineQuality/winequality.csv'
#     task_type='multiclass'
#     preprocess_winequality_regression(data_path,save_path, task_type)
#     print('\n winequality process over!')

# ###output:abalone dataset -https://github.com/gagolews/ordinal-regression-data
#     save_path = '/home/luozhengdong/ordinal_regression/tabR_lzd/data/abalone_classification'# abalone_classification,abalone_regression
#     os.makedirs(save_path, exist_ok=True)
#     # input:
#     data_path = '/home/luozhengdong/datasets/abalone.csv'
#     task_type = 'multiclass'  ##regression,multiclass
#     preprocess_abalone_regression(data_path, save_path, task_type)

# #output:eucalyptus dataset -https://www.kaggle.com/datasets/ishadss/eucalyptus?resource=download
#     save_path = '/home/luozhengdong/ordinal_regression/tabR_lzd/data/eucalyptus_classification' #eucalyptus_regression,eucalyptus_classification
#     os.makedirs(save_path, exist_ok=True)
#     # input:
#     data_path = '/home/luozhengdong/datasets/eucalyptus_lzd.csv'
#     task_type = 'multiclass'  ##regression,multiclass
#     preprocess_eucalyptus_regression(data_path, save_path, task_type)
#     print('eucalyptus over!')

# ##output:user-knowledge dataset -https://www.openml.org/d/1508
#     save_path = '/home/luozhengdong/ordinal_regression/tabR_lzd/data/user_knowledge_regression'
#     os.makedirs(save_path, exist_ok=True)
#     # input:
#     data_path = '/home/luozhengdong/datasets/user-knowledge.arff'
#     task_type = 'regression'  ##regression,multiclass
#     preprocess_user_knowledge_regression(data_path, save_path, task_type)
#     print('user-knowledge over!')

# # output:eye_movements -openml.org/d/1044
#     save_path = '/home/luozhengdong/ordinal_regression/tabR_lzd/data/eye_movements_regression'
#     os.makedirs(save_path, exist_ok=True)
#     # input:
#     data_path = '/home/luozhengdong/datasets/eye_movements.arff'
#     task_type = 'regression'  ##regression,multiclass
#     preprocess_eye_movements_regression(data_path, save_path, task_type)
#     print('eye_movements over!')

# # output:autos-https://www.openml.org/d/9
#     save_path = '/home/luozhengdong/ordinal_regression/tabR_lzd/data/autos_regression'
#     os.makedirs(save_path, exist_ok=True)
#     # input:
#     data_path = '/home/luozhengdong/datasets/dataset_9_autos.arff'
#     task_type = 'regression'  ##regression,multiclass
#     preprocess_autos_regression(data_path, save_path, task_type)
#     print('autos over!')

# # output:cmc-https://www.openml.org/d/23
#     save_path = '/home/luozhengdong/ordinal_regression/tabR_lzd/data/cmc_classification' #cmc_classification,cmc_regression
#     os.makedirs(save_path, exist_ok=True)
#     # input:
#     data_path = '/home/luozhengdong/datasets/dataset_23_cmc.arff'
#     task_type = 'multiclass'  ##regression,multiclass
#     preprocess_cmc_regression(data_path, save_path, task_type)
#     print('cmc over!')

# # output:tae-https://www.openml.org/d/48
#     save_path = '/home/luozhengdong/ordinal_regression/tabR_lzd/data/tae_regression'
#     os.makedirs(save_path, exist_ok=True)
#     # input:
#     data_path = '/home/luozhengdong/datasets/dataset_48_tae.arff'
#     task_type = 'regression'  ##regression,multiclass
#     preprocess_tae_regression(data_path, save_path, task_type)
#     print('tae over!')

# # output:grub_damage-https://www.openml.org/d/338
#     save_path = '/home/luozhengdong/ordinal_regression/tabR_lzd/data/grub_damage_classification' #grub_damage_regression,grub_damage_classification
#     os.makedirs(save_path, exist_ok=True)
#     # input:
#     data_path = '/home/luozhengdong/datasets/phpnYQXoc.arff'
#     task_type = 'multiclass'  ##regression,multiclass
#     preprocess_grub_damage_regression(data_path, save_path, task_type)
#     print('grub_damage over!')

# # output:car-https://www.openml.org/d/1115
#     save_path = '/home/luozhengdong/ordinal_regression/tabR_lzd/data/car_classification' #car_multiclass,car_regression
#     os.makedirs(save_path, exist_ok=True)
#     # input:
#     data_path = '/home/luozhengdong/datasets/php2jDIhh.arff'
#     task_type = 'multiclass'  ##regression,multiclass
#     preprocess_car_regression(data_path, save_path, task_type)
#     print('car over!')
#
# ##autos\car


