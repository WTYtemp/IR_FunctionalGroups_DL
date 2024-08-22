# IR_FunctionalGroups_DL
Paper: Infrared Spectral Analysis for Prediction of Functional Groups Based On Feature Aggregated Deep Learning

## Requirements
* AggMap (bidd-aggmap)
* RDKit
* nistchempy

## 1. /dataset
* ab_IR_gas.csv \
经过筛选后保留的红外光谱文件及其波数采样范围等信息
* dataset_824p_all.csv \
分子官能团标签及红外光谱数据集，光谱特征数为824
* dataset_1647_all.csv \
分子官能团标签及红外光谱数据集，光谱特征数为1647
* dataset_1647_SMILES.csv \
分子SMILES及红外光谱数据集，光谱特征数为1647
* functionalGroupList.csv \
分子官能团标签及相应的SMARTS描述符
* functionalGroupTag.csv \
由RDKit识别的分子官能团标签
* inchi.csv \
气态红外光谱对应的化合物结构描述符InChI
* normData_1647.csv \
MinMax归一化后的红外光谱数据，特征点数为1647

## 2. /importance
* 824p_IR_global_importance_5fold1_seed128 \
用特征维度为824的红外光谱预测官能团，Simply-explainer得到的预测模型中每个官能团的重要性。
* 824p_IR_global_importance_fold1_top10.csv \
从上述文件中筛选出的每个官能团最重要的10处特征波数。

## 3. /model
* {POINTS}_IR_aggmap_correlation_c{CHANNEL NUMBER}.mp \
用correlation距离生成的红外光谱AggMap模型，{POINTS}是特征点数，{CHANNEL NUMBER}是通道数
* 1647_IR_{functional group}_aggmap_correlation_c10.mp \
针对某个官能团子结构的红外光谱生成的10通道AggMap模型
* 824p_IR_MultiLabel_c10_fold{fold number}_seed128.h5 \
用特征维度为824的红外光谱得到的多标签模型，五折交叉

## 4. others
主要为代码文件

* getDataFromNIST.ipynb \
从NIST Chemistry WebBook上下载数据并筛选
* processData.ipynb \
对红外光谱数据点进行归一化等预处理，生成normData_1647.csv
* findFuncGrp.ipynb \
标记分子官能团标签，生成完整的数据集dataset_1647_all.csv
* inchi2smiles.ipynb \
将数据集中化合物的结构转换为SMILES表示，并与相应的红外光谱进行合并，生成dataset_1647_SMILES.csv
* feature_map.ipynb \
训练及可视化AggMap特征图
* train2Class.ipynb \
官能团的二分类预测
* trainMultiLabel.ipynb \
官能团的多标签预测
* trainMultiLabelSub.py \
对主要官能团的子结构进行多标签预测
* interpretMutilLabel.ipynb \
多标签模型的重要性解释，使用Simply-explainer

