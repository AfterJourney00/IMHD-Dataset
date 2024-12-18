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
- ***Nov, 2024:*** 🔈🔈 Download or view online our [videos](https://pan.quark.cn/s/e205d3c0e072) and [segmentations](https://pan.quark.cn/s/a1662d344e4a)!
- ***Sep, 2024:*** 🔈🔈 The instance-level segmentations have been released!
- ***July, 2024:*** 🔈🔈 The raw videos have been released!
- ***May, 2024:*** 🔈🔈 The 32-view 2D & 3D human keypoints have been released!
- ***March, 2024:*** 🎉🎉 [I'M HOI](https://afterjourney00.github.io/IM-HOI.github.io/) is accepted to CVPR 2024!
- ***Jan. 04, 2024:*** 🔈🔈 Fill out [the form](https://forms.gle/3MDh3b4szhFwcYa26) to have access to IMHD$`^2`$!

## Contents
- [Dataset Features](#dataset-features)
- [Dataset Structure](#dataset-structure)
- [Getting Started](#getting-started)
  - [For Windows](#for-windows)
  - [For Ubuntu](#for-ubuntu)
  - [How to use](#how-to-use)
- [FAQs](#faqs)

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
|--object_templates/       # raw and downsampled geometry
|--imu_preprocessed/       # pre-processed IMU signal
|--keypoints2d/            # body keypoints in OP25 format and hand keypoints in MediaPipe format
|--keypoints3d/            # body keypoints in OP25 format and hand keypoints in MediaPipe format
|--video_release/          # raw videos from 32 multiple views
|--mask_release/           # human and object separate segmentations from 32 multiple views
|--ground_truth/           # human motion in SMPL-H format and rigid object motion
|----<date>/
|------<segment_name>/
|--------<sequence_name>/
|----------gt_<part_id>_<start>_<end>.pkl
```
All sub-folders have the similar detailed structure as the shown one of ground truth. Particularly, since motion annotations of some part in some sequence are not ideal, there may exist several `.pkl` files under one sequence folder. To parse the file name meaning of leaf `.pkl` files, here is an example: `gt_0_10_100.pkl: the first motion part which starts from frame_10 and ends at frame_100`.

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
2. Prepare body model. Download [SMPL-H](https://mano.is.tue.mpg.de/login.php) (the extended SMPL+H model) and put the model files under the `body_model/` folder. Overall, the structure of `body_model/` folder should be:
```
body_model/
|--README.md
|--__init__.py
|--body_model.py
|--utils.py
|--smplh/
|----info.txt
|----LICENSE.txt
|----female/
|------model.npz
|----male/
|------model.npz
|----neutral/
|------model.npz
```
3. Run `python visualization.py` to check how to load and visualize IMHD$`^2`$. Results will be saved in `visualizations/`.

## FAQs
**Q1: Which coordinate are the ground-truth motions in? How to align all the motions across different dates?**

*A1: The ground-truth motions are in the **world coordinate** which was calibrated using multi-camera system and may different across dates. To align them, you can use the provided camera parameters in ``calibrations/`` to transform all motion data to camera coordinate.*

**Q2: Which category of object does the motions named with 'bat' in ``20230825/`` and ``20230827/`` interact with?**

*A2: The interacting object category of motions in ``20230825/`` and ``20230827/`` is baseball bat, corresponding to **'baseball'** in the ``object_templates/`` folder.*

**Q3: Which camera serves as the main view?**

*A3: The main view is from the camera labeled with '1'(starting from 0).*

**Q3: How to decode the raw videos to images?**

*A3: Please use the command: `ffmpeg -i <input_path> -qscale:v 2 -f image2 -v error -start_number 0 -threads 64 output/%06d.jpg`*

## Citation
If you find our data or paper helps, please consider citing:
```bibtex
@InProceedings{zhao2024imhoi,
    author    = {Zhao, Chengfeng and Zhang, Juze and Du, Jiashen and Shan, Ziwei and Wang, Junye and Yu, Jingyi and Wang, Jingya and Xu, Lan},
    title     = {I'M HOI: Inertia-aware Monocular Capture of 3D Human-Object Interactions},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {729-741}
}
```

## Acknowledgement
This work was supported by National Key R\&D Program of China (2022YFF0902301), Shanghai Local college capacity building program (22010502800). We also acknowledge support from Shanghai Frontiers Science Center of Human-centered Artificial Intelligence (ShangHAI).

We thank Jingyan Zhang and Hongdi Yang for settting up the capture system. We thank Jingyan Zhang, Zining Song, Jierui Xu, Weizhi Wang, Gubin Hu, Yelin Wang, Zhiming Yu, Xuanchen Liang, af and zr for data collection. We thank Xiao Yu, Yuntong Liu and Xiaofan Gu for data checking and annotations.

## Licenses
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.