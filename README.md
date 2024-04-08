# IMHD$`^2`$: Inertial and Multi-view Highly Dynamic human-object interactions Dataset

[![arXiv](https://img.shields.io/badge/arXiv-2312.08869-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2312.08869) [![project page](https://img.shields.io/badge/project_page-blue)](https://afterjourney00.github.io/IM-HOI.github.io/)
<!-- ![visitor badge](https://visitor-badge.laobi.icu/badge?page_id=AfterJourney00.IMHD-Dataset&left_color=red&right_color=green&left_text=visitors) -->


> **I'M HOI: Inertia-aware Monocular Capture of 3D Human-Object Interactions**  
> *Chengfeng Zhao, Juze Zhang, Jiashen Du, Ziwei Shan, Junye Wang, Jingyi Yu, Jingya Wang, Lan Xu\**

****

<p align="center">
<img src="assets/dataset_gallery_full.png" alt="teaser" width="8128"/>
</p> 

## 🔥News
- ***March, 2024:*** 🎉🎉 [I'M HOI](https://afterjourney00.github.io/IM-HOI.github.io/) is accepted to CVPR 2024!
- ***Jan. 04, 2024:*** 🔈🔈 Fill out [the form](https://forms.gle/3MDh3b4szhFwcYa26) to have access to IMHD$`^2`$!

## Contents
- [Dataset Features](#dataset-features)
- [Dataset Structure](#dataset-structure)
- [Getting Started](#getting-started)
  - [For Windows](#for-windows)
  - [For Ubuntu](#for-ubuntu)
  - [How to use](#how-to-use)

## Dataset Features
IMHD$`^2`$ is featured by:
- Human motion annotation in SMPL-H format, built on [EasyMocap](https://github.com/zju3dv/EasyMocap/tree/master)
- Object motion annotation, built on [PHOSA](https://github.com/facebookresearch/phosa)
- Well-scanned object geometry, using [Polycam](https://poly.cam/)
- Object-mounted IMU sensor measurement, using [Movella DOT](https://www.movella.com/products/wearables/movella-dot)
- 32-view RGB videos & instance-level segmentations, built on [SAM](https://github.com/facebookresearch/segment-anything), [Track-Anything](https://github.com/gaomingqi/Track-Anything) and [XMem](https://github.com/hkchengrex/XMem)
- 32-view 2D&3D human keypoints detection, using [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) and [MediaPipe](https://github.com/google/mediapipe)

## Dataset Structure
```
data/
|--calibrations/           # camera intrinsics and world-to-cam extrinsics
|--objects_template/       # raw and downsampled geometry
|--imu/                    # pre-processed IMU signal
|--keypoints2d/            # body keypoints in OP25 format and hand keypoints in MediaPipe format
|--keypoints3d/            # body keypoints in OP25 format and hand keypoints in MediaPipe format
|--ground_truth/           # human motion in SMPL-H format and rigid object motion
|----<date>/
|------<segment_name>/
|--------<sequence_name>/
|----------gt_<part_id>_<start>_<end>.pkl
```
All the sub-folders have the similar detailed structure as the shown one of ground truth. Particularly, since motion annotations of some part in some sequence are not ideal, there may exist several `.pkl` files under one sequence folder. To parse the file name meaning of leaf `.pkl` files, here is an example: `gt_<0>_<10>_<100>.pkl: the first motion part which starts from frame_10 and ends at frame_100`.

## Getting Started
We tested our code on ``Windows 10``, ``Windows 11``, ``Ubuntu 18.04 LTS`` and ``Ubuntu 20.04 LTS``.

All dependencies:
> python>=3.8  
> CUDA=11.7  
> torch=1.13.0  
> pytorch3d  
> opencv-python  
> matplotlib  
> smplx

### For Windows
```
conda create -n imhd2 python=3.8 -y
conda activate imhd2
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e . --ignore-installed PyYAML
```

### For Ubuntu
```
conda create -n imhd2 python=3.8 -y
conda activate imhd2
conda install --file conda_install_cuda117_pakage.txt -c nvidia
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

### How to use
1. Prepare data. Download IMHD$`^2`$ from [here](https://forms.gle/3MDh3b4szhFwcYa26) and place it under the root directory in the [pre-defined structure](#dataset-structure).
2. Prepare body model. Please refer to [body_model](./body_model/README.md). 
3. Run `python visualization.py` to check how to load and visualize IMHD$`^2`$. Results will be stored in `visualizations/`.

## Citation
If you find our data or paper helps, please consider citing:
```bibtex
@article{zhao2023imhoi,
  title={I'M HOI: Inertia-aware Monocular Capture of 3D Human-Object Interactions},
  author={Zhao, Chengfeng and Zhang, Juze and Du, Jiashen and Shan, Ziwei and Wang, Junye and Yu, Jingyi and Wang, Jingya and Xu, Lan},
  journal={arXiv preprint arXiv:2312.08869},
  year={2023}
}
```

## Acknowledgement
We thank Jingyan Zhang and Hongdi Yang for settting up the capture system. We thank Jingyan Zhang, Zining Song, Jierui Xu, Weizhi Wang, Gubin Hu, Yelin Wang, Zhiming Yu, Xuanchen Liang, af and zr for data collection. We thank Xiao Yu, Yuntong Liu and Xiaofan Gu for data checking and annotations.

## Licenses
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.