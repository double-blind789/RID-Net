# RID-Net: A Hybrid MLP-Transformer network for Robust Point Cloud Registration

PyTorch implementation of the paper:

## Introduction

The robustness of correspondence-based point cloud registration relies on transformation invariance and intrinsic distinctiveness of the descriptors computed for registration. However, for challenging scenarios with different objects having similar local geometry and low point cloud overlap, existing descriptors struggle to retain these properties. This inevitably leads to a considerable loss in matching inlier rates and compromised registration. To address the issue, we propose RID-Net that computes robust **R**otation-**I**nvariant and **D**istinctive descriptors for point cloud registration. Our model works on the philosophy of locally accurate feature abstraction for the points while also accounting for the surroundings. Specifically, it employs a proposed rotation-invariant MLP-based module which is enhanced with high-dimensional positional encoding and partial transformations. The  locally accurate representation of the model is further enhanced by a ring attention mechanism to explicitly focus on the surrounding geometry. The overall network is designed by staking the proposed component, while eventually leveraging coarse-to-fine correspondences and  a robust pose estimator to compute the ultimate output. With extensive evaluation on indoor, outdoor and synthetic point cloud data, we establish the efficacy of the proposed RID-Net. In particular, our method improves the *inlier ratio* by about 4\% on 3DMatch/3DLoMatch dataset and reduces the *rotation and translation errors* by 26.1\% and 19.7\% on the KITTI dataset. 



## Installation

+ Clone the repository:

  ```
  git clone https://github.com/PANFEI-CHENG/RID-Net.git
  cd RID-Net
  ```
+ Create conda environment and install requirements:

  ```
  conda env create -n rid_net python=3.8
  pip install -r requirements.txt
  ```
+ Compile C++ and CUDA scripts:

  ```
  cd cpp_wrappers
  cd pointops
  python setup.py install
  cd ..
  cd ..
  ```

## RUN
  ```
  python main.py configs/3dmatch/demo_config.yaml
  ```
  
 
 ## Reference

 + [GeoTransformer](https://github.com/qinzheng93/GeoTransformer).
 
 + [RoITr](https://github.com/haoyu94/RoITr).
 
 + [CoFiNet](https://github.com/haoyu94/Coarse-to-fine-correspondences). 
 
 + [Point Transformer](https://github.com/POSTECH-CVLab/point-transformer).

 + [RIGA](https://arxiv.org/abs/2209.13252)
 
 We thank the authors for their excellent works!
 