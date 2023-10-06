# PRIOR: Prototype Representation Joint Learning from Medical Images and Reports

## Official repository for the paper [Prototype Representation Joint Learning from Medical Images and Reports, ICCV 2023](https://arxiv.org/abs/2307.12577). 
----

## Introduction
Contrastive learning based vision-language joint pretraining has emerged as a successful representation learning strategy. In this paper, we present a prototype representation learning framework incorporating both global
and local alignment between medical images and reports. In contrast to standard global multi-modality alignment
methods, we employ a local alignment module for finegrained representation. Furthermore, a cross-modality conditional reconstruction module is designed to interchange information across modalities in the training phase by reconstructing masked images and reports. For reconstructing long reports, a sentence-wise prototype memory bank is cons tructed, enabling the network to focus on low-level localized visual and high-level clinical linguistic features. Additionally, a non-auto-regressive generation paradigm is proposed for reconstructing non-sequential reports. Experimental results on five downstream tasks, including supervised classification, zero-shot classification, image-to-text retrieval, semantic segmentation, and object detection, show the proposed method outperforms other state-of-the-art methods across multiple datasets and under different dataset size settings.



### Setup
Run 

```bash
pip install -r requirements.txt
```



###  Data Preparation

#### MIMIC-CXR Dataset

1. Download the Version 2 of the MIMIC-CXR-JPG from `https://physionet.org/content/mimic-cxr-jpg/2.0.0/` to `<image_dataset_path>`

2. Download the reports from MIMIC-CXR `https://physionet.org/content/mimic-cxr/2.0.0/` to `<report_dataset_path>`

3. Run scripts to make json file for pre-training

   

```bash
cd codes/
python prior/data/pretrain/mimiccxr.py build --root_image_path <image_dataset_path> --root_report_path <report_dataset_path> --save_path <dataset_json_path>
```

4. Add `<dataset_json_path>` to `configs/data/mimiccxr.yaml`

```yaml
_target_: prior.data.pretrain.mimiccxr.MimicCxrDataset
dataset_path: <dataset_json_path> # update this line
image_transform: ???
text_transform: ???
num_colors: 1
rate: 1.0
```



### Pre-train

Run

```bash
cd codes/
python scripts/train.py +experiments/pre_train=train_prior
```



### Pre-trained weights
We released our pre-trained model on pretrained/prior_resnet50.pt, you can download image encoder [here](https://github.com/QtacierP/PRIOR/blob/main/pretrained/prior_resnet50.pt). The whole image-text part is released on [Google Drive](http://data1/pujin/released_codes/official/PRIOR/pretrained/total_prior.ckpt).


### Downstream tasks
#### Supervised finetuning
This part is similar to [LOVT](https://github.com/philip-mueller/lovt). The classification model is based on [TorchVision](https://github.com/pytorch/vision); the segmentation model is based on [SMP](https://github.com/qubvel/segmentation_models.pytorch), and the detection model is based on [Lightning-flash](https://github.com/Lightning-Universe/lightning-flash). Since this part has not addtional technical contribution, we do not provide the codes currently. We will update this part in the future.

#### Zero-shot classification
Before running the codes, make sure you have downloaded the [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) dataset in `<root_path>` and downloaded the whole image-text pre-trianed weights of PRIOR from [Google Drive](http://data1/pujin/released_codes/official/PRIOR/pretrained/total_prior.ckpt) to `<pretrained_path>`. 

Add `<root_path>` and `<dataset_path>` to `configs/data/chexpert_zero_shot.yaml`

```yaml
_target_: prior.data.zero_shot_classification.chexpert_zls.CheXPertZeroClsDataset
dataset_path: ??? # update this line
transform: ???
num_colors: 1
root_path: ??? # update this line
rate: 1.0
```
Add `<pretrained_path>` to `configs/experiments/zero_shot_classification/test_prior.yaml`

```yaml
zero_shot_classification_model:
    ......
    pre_trained_path: # update this line
    ......
``````

Then run

```bash
cd codes/
python scripts/downstream.py +experiments/zero_shot_classification=test_prior
```


### Acknowledgement
Some of the code is borrowed from [LOVT](https://github.com/philip-mueller/lovt), [GLoRIA](https://github.com/marshuang80/gloria). Thanks for their great work.


### Citation
If you find this work useful in your research, please cite:
```
@inproceedings{PRIOR,
  title={PRIOR: Prototype Representation Joint Learning from Medical Images and Reports},
  author={Cheng, Pujin and Lin, Li and Lyu, Junyan and Huang, Yijin and Luo, Wenhan and Tang, Xiaoying},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```
