# PRIOR: Prototype Representation Joint Learning from Medical Images and Reports

This is the implementation of the ICCV2023 paper with id 9929 ***PRIOR: Prototype Representation Joint Learning from Medical Images and Reports.***



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
We released our pre-trained model on pretrained/prior_resnet50.pt, you can download [here](https://github.com/QtacierP/PRIOR/blob/main/pretrained/prior_resnet50.pt)


### Downstream tasks
This part is under code review and will be released soon (In August 2023).

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
