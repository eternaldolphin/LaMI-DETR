# LaMI-DETR

This repository is the official implementation of the **ECCV 2024** paper **LaMI-DETR: Open-Vocabulary Detection with Language Model Instruction**.

[![arXiv](https://img.shields.io/badge/Arxiv-2407.11335-b31b1b.svg?logo=arXiv)](https://arxiv.org/pdf/2407.11335)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lami-detr-open-vocabulary-detection-with/open-vocabulary-object-detection-on-lvis-v1-0)](https://paperswithcode.com/sota/open-vocabulary-object-detection-on-lvis-v1-0?p=lami-detr-open-vocabulary-detection-with)

## Installation
The code is tested under python=3.9 torch=1.10.0 [cuda=11.7](https://drive.google.com/file/d/1A57019pFuRRjaQAVAv_lfeWWBWadfcgE/view?usp=sharing). Please [download](https://drive.google.com/file/d/1nIq4gAHvNYSaC_dnozVtHeTH0Rsw-DHY/view?usp=sharing) and unzip this environment under your conda envs dir.
```shell
cd your_conda_envs_path
unzip tar -xvf lami.tar
vim your_conda_envs_path/lami/bin/pip
change '#!~/.conda/envs/lami/bin/python' to '#!your_conda_envs_path/lami/bin/python'
export CUDA_HOME=/usr/local/cuda-11.7
```

or you can create a conda environment and activate it. Install `PyTorch` following the [official documentation](https://pytorch.org/).
For example,
```shell
conda create -n lami python=3.9
conda activate lami
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
export CUDA_HOME=/usr/local/cuda-11.7
```

Check the torch installation.
```shell
python
>>> import torch
>>> torch.cuda.is_available()
True
>>> from torch.utils.cpp_extension import CUDA_HOME
>>> CUDA_HOME
'/usr/local/cuda-11.7'
>>> exit()
```

Install the [detectron2](https://github.com/facebookresearch/detectron2) and [detrex](https://github.com/IDEA-Research/detrex).
```shell
cd LaMI-DETR
pip install -e detectron2
pip install -e .
```


## Preparation

### Datasets
Download the [MS-COCO](https://cocodataset.org/#download) dataset to `dataset/coco`.
Download and unzip the [LVIS annotation](https://drive.google.com/file/d/1k-o3Dxsj_qAvzR7bszrMoaY0DOwRZfEX/view?usp=sharing) to `dataset/lvis`.
```text
LaMI-DETR/dataset   
├── coco/ 
│   ├── train2017/
│   └── val2017/
├── lvis
|   ├── lvis_v1_train_norare.json
|   ├── lvis_v1_val.json
|   ├── lvis_v1_minival.json
|   ├── lvis_v1_train_norare_cat_info.json
|   ├── lvis_v1_seen_classes.json
|   └── lvis_v1_all_classes.json
├── cluster
|   ├── lvis_cluster_128.npy
|   └── vg_cluster_256.npy
└── metadata 
    ├── lvis_visual_desc_convnextl.npy
    ├── lvis_visual_desc_confuse_lvis_convnextl.npy
    └── concept_dict_visual_desc_convnextl.npy

```

### Custom Dataset Register
Referring to Detectron2  
```text
detectron2/detectron2/data/datasets/builtin.py  
detectron2/detectron2/data/datasets/builtin_meta.py  
```

### Custom Concept Prediction
```text
Change "model.eval_query_path" in config file
```

### Pretrained Models
```text
LaMI-DETR/pretrained_models   
├── idow_convnext_large_12ep_lvis/ 
│   └── model_final.pth
├── idow_convnext_large_12ep_vg/
│   └── model_final.pth
├── lami_convnext_large_obj365_12ep.pth
├── clip_convnext_large_trans.pth
└── clip_convnext_large_head.pth
```
[clip_convnext_large_head.pth](https://drive.google.com/file/d/1Jr9mfnAyiSGjsh09X0pqspuagsMyzHSS/view?usp=sharing)


## Inference  
In the paper, we reported p2 layer score ensemble results. This repository provides p3 layer results, which are generally higher. We found p2 and p3 layers with ConvNeXt yield similar results, but p3 is much faster. Thus, we recommend using p3.

|  #  |  Training Data  |  Inference Data  |   AP   |   APr   |                                Script                                | Init checkpoint | Checkpoint |
|:---:|:---------------:|:----------------:|:------:|:-------:|:--------------------------------------------------------------------:|:----------:|:----------:|
|  1  |    LVIS-base    |        LVIS      |  41.6  |   43.3  |  [script](lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py)   | [clip_convnext_large_trans.pth](https://drive.google.com/file/d/1jwWmB80oJi2x8YBZTeSX_pgMggsi67GJ/view?usp=sharing)  | [idow_convnext_large_12ep_lvis/model_final.pth](https://drive.google.com/file/d/1DRIYuaW4oV_ghFLRX2VG-cALWsF0fxyk/view?usp=sharing)  |
|  2  |    VGdedup      |        LVIS      |  35.4  |   38.8  | [script](lami_dino/configs/dino_convnext_large_4scale_12ep_vg.py) | [lami_convnext_large_obj365_12ep.pth](https://drive.google.com/file/d/12OsjOCapOiRsOlTx1YYeJ3PkSiH_zRrK/view?usp=sharing)  | [idow_convnext_large_12ep_vg/model_final.pth](https://drive.google.com/file/d/1p5ROlpMeLml4Nns2sByv80RO8F4QVgUn/view?usp=sharing)  |


OV-LVIS 
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py --num-gpus 4 --eval-only train.init_checkpoint=pretrained_models/idow_convnext_large_12ep_lvis/model_final.pth
```
Zero-shot LVIS 
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_vg.py --num-gpus 4 --eval-only train.init_checkpoint=pretrained_models/idow_convnext_large_12ep_vg/model_final.pth
```
For a quick debug you can update numpy to 1.24.0 and install lvis-debug, then comment the 372 line and uncomment the 373 line in detectron2/detectron2/evaluation/lvis_evaluation.py 
```
pip uninstall lvis
git clone https://github.com/eternaldolphin/lvis-debug.git
cd lvis-debug
pip install -e .
cd ../
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py --num-gpus 1 --ddebug --eval-only
```


## Training
OV-LVIS 
```bash
python tools/train_net.py --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py --num-gpus 8 train.init_checkpoint=pretrained_models/clip_convnext_large_trans.pth
```
Zero-shot LVIS 
```bash
python tools/train_net.py --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_vg.py --num-gpus 8 train.init_checkpoint=pretrained_models/lami_convnext_large_obj365_12ep.pth
```
For a quick debug you can update numpy to 1.24.0 and install lvis-debug, then comment the 372 line and uncomment the 373 line in detectron2/detectron2/evaluation/lvis_evaluation.py 
```bash
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py --num-gpus 1 --ddebug
```


## TODO List
- [x] Release inference codes.
- [x] Release checkpoints.
- [x] Release training codes.
- [ ] Release demo.
- [ ] Release coco and o365 inference codes.


## 🤝🏼 Cite Us

```
@inproceedings{du2024lami,
  title={LaMI-DETR: Open-Vocabulary Detection with Language Model Instruction},
  author={Du, Penghui and Wang, Yu and Sun, Yifan and Wang, Luting and Liao, Yue and Zhang, Gang and Ding, Errui and Wang, Yan and Wang, Jingdong and Liu, Si},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  year={2024}
}
```

## 💖 Acknowledgement
<span id="acknowledgement"></span>
LaMI-DETR is built based on [detectron2](https://github.com/facebookresearch/detectron2) and [detrex](https://github.com/IDEA-Research/detrex), thanks to all the contributors!

