# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.metrics import auc as calculate_auc

import matplotlib.pyplot as plt
import seaborn as sns

from aggmap import AggMap, AggMapNet, loadmap
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
physical_gpus = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_gpus[0], True)

np.random.seed(666)


def prc_auc_score (y_true, y_score):
    precision, recall, threshold = precision_recall_curve(y_true, y_score) # PRC_AUC
    auc = calculate_auc(recall, precision)
    return auc


dataset = './dataset/dataset_1647_all.csv'
data_df = pd.read_csv(dataset)
main_func = 'alkane' # alkane, alkene, alcohols, amines, amides
data_df = data_df[data_df[main_func] == 1]
func_grps = [ '-CH-', '-CH2-', '-CH3', '-CH(CH3)2', '-C(CH3)3']   # alkane
# func_grps = ['-CH=CH2', '-CH=CH-', '-C=CH2', '-C=CH-', '-C=C-']   # alkene
# func_grps = ['primary alcohols', 'secondary alcohols', 'tertiary alcohols']   # alcohols
# func_grps = ['primary amines', 'secondary amines', 'tertiary amines']   # amines
# func_grps = ['primary amides', 'secondary amides', 'tertiary amides']   # amides

dfx = data_df[data_df.columns[1:-42]]
dfy = data_df[func_grps]

X = dfx.values
Y = dfy.values.astype(float)

channels = 10
mp = loadmap('./model/1647_IR_{}_aggmap_correlation_c{}.mp'.format(main_func, channels))


results = {}
results_num = {}
for func in func_grps:
    results[func] = []
    results_num[func] = []


for random_seed in [128]:   # 随机种子用于划分数据集
    outer = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    fold_idx = 0
    outer_split = outer.split(X)
    
    for train_idx, test_idx in outer_split: # 五折交叉
        train_X, test_X = X[train_idx], X[test_idx]
        trainY, testY = Y[train_idx], Y[test_idx]
        
        trainX = mp.batch_transform(train_X, scale_method='standard')
        testX = mp.batch_transform(test_X, scale_method='standard')
        print("trainX shape is: " + str(trainX.shape))
        print("testX shape is: " + str(testX.shape))
        
        clf = AggMapNet.MultiLabelEstimator(epochs=300, batch_size=4, dense_layers=[256, 128], dropout=0.1, batch_norm=True)
        clf.fit(trainX, trainY)

        # clf._model.save('./model/1647_IR_MultiLabel_c{}_fold{}_seed128.h5'.format(channels, fold_idx))
        
        print('Training finished.')
        y_true = testY
        y_pred = clf.predict(testX)
        y_score = clf.predict_proba(testX)
        
        for i in range(len(func_grps)):
            tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred[:, i]).ravel()
            
            acc = (tp + tn) / sum([tn, fp, fn, tp])
            sensitivity = tp / sum([tp, fn])
            specificity = tn / sum([tn, fp])

            # prc_auc = prc_auc_score(y_true[i], y_score[i])
            # roc_auc = roc_auc_score(y_true[i], y_score[i])

            precision = tp / sum([tp, fp])
            recall = sensitivity
            F1 = 2 * precision * sensitivity / (precision + sensitivity)
        
            # res 记录结果用来画图
            res = clf.history   # dictionary
            res['fold'] = fold_idx
            res['channel'] = channels
            res['random_seed'] = random_seed

            # res_num 记录结果用来看数值
            fold_num = "fold_%s" % str(fold_idx).zfill(2)
            res_num = {'fold': fold_num,
                  'random_seed': random_seed,
                  'accuracy': acc,
                  #'prc_auc': prc_auc,
                  #'roc_auc': roc_auc,
                  'sensitivity': sensitivity,
                  'specificity': specificity,
                  'precision': precision,
                  'recall': recall,
                  'F1': F1}

            results[func_grps[i]].append(res)
            results_num[func_grps[i]].append(res_num)
        
        fold_idx += 1


result_path = './result/aggmapnet_1647_{}_c{}.csv'.format(main_func, channels)
res_num_all = []

for func in func_grps:
    path_num = './result/sub_group_test/{}_test_1647_{}_c{}.csv'.format(main_func, func, channels)
    # path = './result/result_valid_1647_c1/result_1647_{}.csv'.format(func)
    
    res_num_df = pd.DataFrame(results_num[func])
    res_num_df.to_csv(path_num)
    # res_df = pd.DataFrame(results[func])
    # res_df.to_csv(path)
    
    res_mean = res_num_df.groupby('random_seed').apply(np.mean).mean().round(3)
    #res_std = res_num_df.groupby('random_seed').apply(np.std).mean().round(3)

    res_num_all.append([func] + res_mean[['accuracy', 'F1']].tolist())

res_num_all = pd.DataFrame(res_num_all, columns=['func', 'accuracy', 'F1'])
res_num_all.to_csv(result_path, index=False)


color = sns.color_palette("rainbow_r", 6) #PiYG
sns.palplot(color)

for i in [0]:
    res_df = pd.DataFrame(results[func_grps[i]])

    sns.set(style='white', font_scale=2)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18,16), sharex=False, sharey=False)
    ax0, ax1, ax2, ax3 = axes.ravel()

    col = 'roc_auc'
    acc_mean = res_df.groupby(['channel']).agg({col: lambda x:x.tolist()})[col].apply(lambda x:np.array(x).mean(axis=0)).apply(pd.Series).T
    acc_mean.plot(ax=ax0, lw=4, color=color)
    ax0.set_xlabel('Epochs')
    ax0.set_ylabel('Train Roc_Accuracy')

    col = 'loss'
    acc_mean = res_df.groupby(['channel']).agg({col: lambda x:x.tolist()})[col].apply(lambda x:np.array(x).mean(axis=0)).apply(pd.Series).T
    acc_mean.plot(ax=ax1, lw=4, color=color)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Loss')

    col = 'val_roc_auc'
    acc_mean = res_df.groupby(['channel']).agg({col: lambda x:x.tolist()})[col].apply(lambda x:np.array(x).mean(axis=0)).apply(pd.Series).T
    acc_mean.plot(ax=ax2, lw=4, color=color)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Validation Roc_Accuracy')

    col = 'val_loss'
    acc_mean = res_df.groupby(['channel']).agg({col: lambda x:x.tolist()})[col].apply(lambda x:np.array(x).mean(axis=0)).apply(pd.Series).T
    acc_mean.plot(ax=ax3, lw=4, color=color)
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Validation Loss')

    fig.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3)
    plt.savefig('./result/1647_{}_correlation_valid.png'.format(func_grps[i]), bbox_inches='tight', dpi=400)
