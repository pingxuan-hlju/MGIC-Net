# MGIC-Net

## Introduction  

The project  is an implementation of Multimodal Graph Intra- and Cross-Attention Network for Disease Classification with Joint-task Optimization (MGIC-Net). 

---

## Catalogs  

- **/classandseg/MGIC_Net**: Contains the code implementation of MMDMA algorithm.
- **/MGIC_Net/network_architecture/generic_UNet.py**: Defines the segmentation model.
- **/classandseg/MGIC_Net/models/mv3Dunet_down_text_cmsa_small.py**: Defines the classification model.
- **/classandseg/MGIC_Net/models/mv3Dunet_down_text_cmsa_mid.py**: Defines the classification model.
- **/classandseg/MGIC_Net/run/run_training.py**: Train the model.
- **/classandseg/MGIC_Net/inference/predict.py**: test the model.

---

## Environment  

The MGIC-Net code has been implemented and tested in the following development environment: 

- torch==2.0.1
- tqdm==4.66.1
- dicom2nifti==2.4.8
- scikit-image==0.21.0
- medpy==0.4.0
- scipy==1.11.2
- batchgenerators==0.25
- numpy==1.21.6
- scikit-learn
- SimpleITK
- pandas
- requests
- nibabel
- tifffile
- matplotlib
- wandb==0.15.8

---

## How to Run the Code  

1. **Train the model**: 

   ```bash
   python classandseg/MGIC_Net/run/run_training.py
   ```

2. **Test the model:**

   ```bash
   python classandseg/MGIC_Net/inference/predict.py
   ```
