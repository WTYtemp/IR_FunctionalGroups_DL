# IR_FunctionalGroups_DL
Paper: Infrared Spectral Analysis for Prediction of Functional Groups Based On Feature Aggregated Deep Learning

## Requirements
* AggMap [(bidd-aggmap)](https://github.com/shenwanxiang/bidd-aggmap)
* RDKit
* nistchempy

## 1. /dataset
* ab_IR_gas.csv \
IR spectra files retained after filtering and their wavenumber sampling ranges and other information
* dataset_824p_all.csv \
Molecular functional group labels and IR spectra dataset, spectral feature count: 824
* dataset_1647_all.csv \
Molecular functional group labels and IR spectra dataset, spectral feature count: 1647
* dataset_1647_SMILES.csv \
Molecular SMILES and IR spectra dataset, spectral feature count: 1647
* functionalGroupList.csv \
Molecular functional group labels and corresponding SMARTS descriptors
* functionalGroupTag.csv \
Molecular functional group labels recognized by RDKit
* inchi.csv \
InChI descriptors corresponding to gas-phase IR spectra
* normData_1647.csv \
IR spectra data after MinMax normalization, spectral feature count: 1647

## 2. /importance
* 824p_IR_global_importance_5fold1_seed128 \
Using IR spectra (824 features) to predict functional groups, the importance of each functional group in the prediction model obtained by Simply-explainer.
* 824p_IR_global_importance_fold1_top10.csv \
The 10 most important characteristic wavenumber positions for each functional group obtained from the above file.

## 3. /model
* {POINTS}_IR_aggmap_correlation_c{CHANNEL NUMBER}.mp \
IR spectra AggMap model generated using correlation distance, {POINTS} is the count of feature points, {CHANNEL NUMBER} is the number of channels
* 1647_IR_{functional group}_aggmap_correlation_c10.mp \
10-channel AggMap model generated for IR spectra of a certain {functional group}
* 824p_IR_MultiLabel_c10_fold{fold number}_seed128.h5 \
Multi-label model trained on IR spectra (824 features), five-fold cross-validation

## 4. others
Mainly code files.

* getDataFromNIST.ipynb \
Download data from NIST Chemistry WebBook and filter.
* processData.ipynb \
Perform normalization and other preprocessing on IR spectra data, generate normData_1647.csv.
* findFuncGrp.ipynb \
Assign molecular functional group labels, generate the complete dataset dataset_1647_all.csv.
* inchi2smiles.ipynb \
Convert the structures of compounds in the dataset to SMILES strings, and merge them with corresponding IR spectra, generate dataset_1647_SMILES.csv.
* feature_map.ipynb \
Train and visualize AggMap feature maps.
* train2Class.ipynb \
Binary classification prediction of functional groups.
* trainMultiLabel.ipynb \
Multi-label prediction of functional groups.
* trainMultiLabelSub.py \
Perform multi-label prediction on substructures of major functional groups.
* interpretMutilLabel.ipynb \
Importance interpretation of trained multi-label models, using Simply-explainer.
