import json
from pathlib import Path
import os
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(last,ax, y_true, y_pred, classes,title):
    labels, counts = np.unique(y_true, return_counts=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im =ax.imshow(cm_normalized, interpolation='nearest', cmap='YlOrBr')
    ax.set_title(title,fontweight='bold')
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    # if last=='false':
    #     cbar = fig.colorbar(im, ax=ax)
    #     cbar.ax.set_yticklabels([])
    # 遍历每个单元格，添加文本（案例个数和百分比）
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cell_text = []
            if cm[i, j] > 0:
                num_pred=cm[i, j]
                cell_text.append(f"{num_pred:d}")  # 案例个数
                num_true=counts[i]
                cell_text.append(f"{(num_pred/ num_true * 100):.1f}%")  # 百分比
            else:
                # cell_text.append("0")
                cell_text.append("0%")
            ax.text(j, i, '\n'.join(cell_text),
                    va='center',
                    ha='center',
                    color='black' if cm[i, j] > thresh else 'black')

    ax.set_ylabel('y_true',fontweight='bold')
    ax.set_xlabel('y_pred',fontweight='bold')


def plot_confusion_matrix2(y_true,y_pred,title,save_path):
    labels, counts = np.unique(y_true, return_counts=True)
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    # plt.title(title)
    # 绘制混淆矩阵
    fig, ax = plt.subplots(figsize=(4,3))
    im = ax.imshow(cm, interpolation='nearest', cmap='YlOrBr')  # 使用'YlOrBr'颜色映射
    # ax.figure.colorbar(im, ax=ax)
    # cbar = fig.colorbar(im, ax=ax)
    # cbar.ax.set_yticklabels([])

    # 设置x轴和y轴的标签
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           # title='Confusion Matrix',
           ylabel='y_true',
           xlabel='y_pred')

    # 旋转x轴标签以适应图表
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor") #rotation=45

    # # 遍历每个单元格，添加文本（案例个数和百分比）
    # thresh = cm.max() / 2.
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         cell_text = []
    #         if cm[i, j] > 0:
    #             num_pred=cm[i, j]
    #             cell_text.append(f"{num_pred:d}")  # 案例个数
    #             num_true=counts[i]
    #             cell_text.append(f"({(num_pred/ num_true * 100):.2f}%)")  # 百分比
    #         else:
    #             cell_text.append("0")
    #             cell_text.append("(0%)")
    #         ax.text(j, i, '\n'.join(cell_text),
    #                 va='center',
    #                 ha='center',
    #                 color='white' if cm[i, j] > thresh else 'black')
    plt.savefig(save_path, format='png',dpi=200)
    fig.tight_layout()
    plt.show()
##data1
y_true1=np.load('/home/luozhengdong/ordinal_regression/tabR_lzd/data/wine_quality_classification/Y_test.npy')#
base_path = '/home/luozhengdong/ordinal_regression/tabR_lzd/exp/tabr_classGroup/acc/wine_quality-evaluation/'#315eucalyptus,abalone,wine_quality,310car,711cmc
save_path='/home/luozhengdong/ordinal_regression/tabR_lzd/exp/tabr_classGroup/wine_quality-tabcgok_tabr.png'
scores = []
for seed in range(15):
    x=base_path+str(seed)+'/report.json'
    acc=json.loads(Path(x).read_text())['metrics']['test']['score']
    scores.append(acc)
max_index = scores.index(max(scores))

file_path = '{}/predictions.npz'.format(max_index)
full_path = os.path.join(base_path, file_path)
predictions = np.load(full_path)['test']
y_pred1=scipy.special.softmax(predictions, axis=1).argmax(axis=1).astype(np.int64)
print('obtain y_pred and y_ture')
##data2
y_true2=np.load('/home/luozhengdong/ordinal_regression/tabR_lzd/data/wine_quality_classification/Y_test.npy')#
base_path = '/home/luozhengdong/ordinal_regression/tabR_lzd/exp/tabr/acc/wine_quality-evaluation/'
# y_true2=np.load('/home/luozhengdong/ordinal_regression/tabR_lzd/data/wine_quality_classification/Y_test.npy')#
# base_path = '/home/luozhengdong/ordinal_regression/tabR_lzd/exp/tabr_classGroup/acc/wine_quality-evaluation/'#315eucalyptus,abalone,wine_quality,310car,711cmc
# save_path='/home/luozhengdong/ordinal_regression/tabR_lzd/exp/tabr_classGroup/wine_quality_confusion_matrix.png'
scores = []
for seed in range(15):
    x=base_path+str(seed)+'/report.json'
    acc=json.loads(Path(x).read_text())['metrics']['test']['score']
    scores.append(acc)
max_index = scores.index(max(scores))

file_path = '{}/predictions.npz'.format(max_index)
full_path = os.path.join(base_path, file_path)
predictions = np.load(full_path)['test']
y_pred2=scipy.special.softmax(predictions, axis=1).argmax(axis=1).astype(np.int64)
# plot_confusion_matrix2(y_true,y_pred,'confusion_matrix',save_path)
# 创建一个一行两列的子图网格
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))




# 获取唯一的类别标签
classes1 = unique_labels(y_true1, y_pred1)
# 在第一个子图上绘制第一个混淆矩阵
plot_confusion_matrix('false',axs[0], y_true1, y_pred1, classes=classes1,
                      title='TabCGOK',)
classes2 = unique_labels(y_true2, y_pred2)
# 在第二个子图上绘制第二个混淆矩阵
plot_confusion_matrix('true',axs[1], y_true2, y_pred2, classes=classes2,
                      title='backbone', )


# 显示图形

fig.tight_layout()
plt.savefig(save_path, format='png', dpi=200)
plt.show()

print('confusion_matrix over!')