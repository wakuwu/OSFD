# Transferable Adversarial Attacks for Object Detection using Object-Aware Significant Feature Distortion

---
This is an official implementation of the **OSFD** adversarial attack method code.

## Installation

---
### 1. Prepare Environment
```shell
# Create a conda environment
conda create -n OSFD python=3.10 -y

# Install an appropriate version of PyTorch
conda activate OSFD
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# Install OpenMMLab's openmim
pip install -U openmim

# Install mmcv and mmdet using openmim
mim install "mmcv-full==1.7.1"
mim install "mmdet==2.28.2"

# Install other libs
pip install imgaug
pip install tensorboard

```

### 2. Prepare Dataset
```shell
# Download VOC dataset: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html
cd OSFD/data
wget --no-check-certificate http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar

# Construct a subset for attack
(->OSFD) cd ..
conda activate OSFD
python tools/datasets/split_dataset.py --num 2000 --voc_type "VOC2012" --paper True

# Covert VOC dataset to COCO type
python tools/datasets/voc12_to_coco.py --num 2000 --img_size 800 --voc_type "VOC2012"
```

### 3. Prepare Models
```shell
cd OSFD/ummdet/checkpoints/models/
conda activate OSFD
# Download model checkpoints and configs
mim download mmdet --config yolov3_d53_mstrain-608_273e_coco --dest .
mim download mmdet --config yolof_r50_c5_8x8_1x_coco --dest .
mim download mmdet --config yolox_l_8x8_300e_coco --dest .
mim download mmdet --config vfnet_r50_fpn_mstrain_2x_coco --dest .
mim download mmdet --config fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco --dest .
mim download mmdet --config faster_rcnn_r101_caffe_fpn_mstrain_3x_coco --dest .
mim download mmdet --config mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco --dest .
mim download mmdet --config detr_r50_8x2_150e_coco --dest .

# Modify model configs for attack
# ummdet
#    └─checkpoints
#            ├─train_cfg:       The white-box model configs to be attacked.
#            ├─eval_cfg:        White box and black box configs for all models.
#            └─models:          Model files with the '.pth' suffix.

```



## Attack

---
All the attack configuration is in the yaml file, you can refer to base.yaml to customize it yourself. 

**model_name** can be: `yolov3`, `vfnet`, `faster_rcnn`, `swin`.

Before the attack, you need to set OSFD (project_path) for attack_{model_name}.yaml files.
```shell
# Main attack script
(->OSFD) cd ../../..
python run_perturb.py config/attack_{model_name}.yaml
```