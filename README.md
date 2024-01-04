# IMHD$`^2`$: Inertial and Multiview Highly Dynamic Human-object Dataset

[![arXiv](https://img.shields.io/badge/Arxiv-2312.08869-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2312.08869) [![project page](https://img.shields.io/badge/project_page-blue)](https://afterjourney00.github.io/IM-HOI.github.io/)
<!-- ![visitor badge](https://visitor-badge.laobi.icu/badge?page_id=AfterJourney00.IMHD-Dataset&left_color=red&right_color=green&left_text=visitors) -->


> **I'M HOI: Inertia-aware Monocular Capture of 3D Human-Object Interactions**  
> *Chengfeng Zhao, Juze Zhang, Jiashen Du, Ziwei Shan, Junye Wang, Jingyi Yu, Jingya Wang, Lan Xu\**

****

<p align="center">
<img src="assets/dataset_gallery_full.png" alt="teaser" width="8128"/>
</p> 

## 🔥News
- ***Jan. 04, 2024:*** Fill out [the form]() to download dataset!

## Contents
- [Getting Started](#getting-started)
  - [For Windows](#for-windows)
  - [For Ubuntu](#for-ubuntu)
- [Dataset Features](#dataset-features)
- [Dataset Structure](#dataset-structure)
- [Example usage](#example-usage)
  - [Visualize IMU signals](#visualize-imu-signals)
  - [Visualize Ground-Truth motion](#visualize-ground-truth-motion)

## Getting Started
We tested our code on ``Windows 10``, ``Windows 11``, ``Ubuntu 18.04 LTS`` and ``Ubuntu 20.04 LTS``.

All dependencies:
> python>=3.8  
> CUDA=11.7  
> torch=1.13.0  
> pytorch3d  
> neural_renderer

### For Windows
```
conda create -n imhd2 python=3.8 -y
conda activate imhd2
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e . --ignored-install PyYAML
cd ../
git clone https://github.com/JiangWenPL/multiperson.git && cd neural_renderer
python setup.py install
```

### For Ubuntu
```
conda create -n imhd2 python=3.8 -y
conda activate imhd2
conda install --file conda_install_cuda117_pakage.txt
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
git clone https://github.com/JiangWenPL/multiperson.git && cd neural_renderer
python setup.py install
```
It is worth noting that we use the fast version of `neural_renderer` in [this repo](https://github.com/JiangWenPL/multiperson/tree/master/neural_renderer). You may have to modify the `.cu` files according to [this issue](https://github.com/daniilidis-group/neural_renderer/issues/144).

## Dataset Features
IMHD$^2$ is featured by:
- Human motion annotation in SMPL-H format, built on [EasyMocap](https://github.com/zju3dv/EasyMocap/tree/master)
- Object motion annotation, built on [PHOSA](https://github.com/facebookresearch/phosa)
- Well-scanned object geometry, using [Polycam](https://poly.cam/)
- Object-mounted IMU sensor measurement, using [Movella DOT](https://www.movella.com/products/wearables/movella-dot)
- 32-view RGB videos & instance-level segmentations, built on [SAM](https://github.com/facebookresearch/segment-anything), [Track-Anything](https://github.com/gaomingqi/Track-Anything) and [XMem](https://github.com/hkchengrex/XMem)
- 32-view 2D&3D human keypoints detection, using [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) and [MediaPipe](https://github.com/google/mediapipe)

## Dataset Structure
```
IMHD2/
|--calibrations/           # camera intrinsics and world-to-cam extrinsics
|--objects_template/       # raw and downsampled geometry
|--imu/                    # pre-processed IMU signal
|--keypoints2d/            # body keypoints in OP25 format and hand keypoints in MediaPipe format
|--keypoints3d/            # body keypoints in OP25 format and hand keypoints in MediaPipe format
|--ground_truth/           # human motion in SMPL-H format and rigid object motion
|----<date>/
|------<segment_name>/
|--------<sequence_name>/
|----------xxxxxx.pkl
```
All the subfolders have the same structure as the shown one of ground truth.

## Example usage
Coming soon...
<!-- Here we describe some example usages of our dataset: 

### Visualize IMU signals

We provide sample code in `shanzw1.py` to visualize IMU signals in a graph. Run with:
```
python shanzw1.py ...... 
```
 

### Visualize Ground-Truth motion

We provide example code in `shanzw2.py` to visualize our ground-truth data. Once you have the dataset and dependencies ready, run:
```
python shanzw2.py ...... 
```
you should be able to see the video.  -->

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
We thank Jingyan Zhang and Hongdi Yang for settting up the capture system. We thank Jingyan Zhang, Zining Song, Jierui Xu, Weizhi Wang, Gubin Hu, Yelin Wang, Zhiming Yu, Xuanchen Liang, af and zr for data collection. We thank Xiao Yu, Yuntong Liu, Xiaofan Gu for data checking and annotations.

## Licenses
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.