# NeTO

### MindSpore implement of

NeTO: Neural Reconstruction of Transparent Objects with Self-Occlusion Aware Refraction-Tracing (ICCV 2023)

## [Project Page](https://www.xxlong.site/NeTO/) | [Paper](https://arxiv.org/pdf/2303.11219.pdf) | [Data](https://drive.google.com/drive/folders/1gSLI58O8FRN_Dq_Zjv6z3W2jfvqnE7Jo?usp=drive_link)

<!-- we will release the code soon. -->

![](./docs/images/teaser.png)

## Introduction

 We present a novel method called NeTO, for capturing the 3D geometry of solid transparent objects from 2D images via volume rendering. 
    Reconstructing transparent objects is a very challenging task, which is ill-suited for general-purpose reconstruction techniques due to the specular light transport phenomena.
    Although existing refraction-tracing-based methods, designed especially for this task, achieve impressive results, they still suffer from unstable optimization and loss of fine details since the explicit surface representation they adopted is difficult to be optimized, and the self-occlusion problem is ignored for refraction-tracing.
    In this paper, we propose to leverage implicit Signed Distance Function (SDF) as surface representation and optimize the SDF field via volume rendering with a self-occlusion aware refractive ray tracing. 
    The implicit representation enables our method to be capable of reconstructing high-quality reconstruction even with a limited set of views, and the self-occlusion aware strategy makes it possible for our method to accurately reconstruct the self-occluded regions. 
    Experiments show that our method achieves faithful reconstruction results and outperforms prior works by a large margin.

#### Data Convention

The data is organized as follows:

```
<case_name>
|-- cameras_xxx.npz    # camera parameters
|-- screen_point.npy   # 3D position on the screen 
|-- mask
    |-- 000.png        # target mask each view (For unmasked setting, set all pixels as 255)
    |-- 001.png
    ...
|-- light_mask
    |-- 000.png        # ray-position correspondences mask each view (For unmasked setting, set all pixels as 255)
    |-- 001.png
    ...  
```

Here the `cameras_xxx.npz` follows the data format in [IDR](https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md), where `world_mat_xx` denotes the world to image projection matrix, and `scale_mat_xx` denotes the normalization matrix.

### Dependencies

- mindspore==2.1.1
- trimesh==3.9.8 
- pyhocon==0.3.57
- icecream==2.1.0
- PyMCubes==0.1.2

### Running

- **Pretrained Data**

4 pretrained MindSpore models are available at [release](https://github.com/nauyihsnehs/NeTO-MindSpore/releases).

- **Training**

```shell
python NeTO.py --mode train --conf ./confs/base9.conf --case <case_name>
```

- **Testing** 

```shell
python NeTO.py --mode test --conf <config_file> --case <case_name>
```

The corresponding mesh can be found in `exp/<case_name>/<exp_name>/meshes/<iter_steps>.ply`.

## Citation

Cite as below if you find this repository is helpful to your project:

```
@article{li2023neto,
  title={NeTO: Neural Reconstruction of Transparent Objects with Self-Occlusion Aware Refraction-Tracing},
  author={Li, Zongcheng and Long, Xiaoxiao and Wang, Yusen and Cao, Tuo and Wang, Wenping and Luo, Fei and Xiao, Chunxia},
  journal={arXiv preprint arXiv:2303.11219},
  year={2023}
}
```

## Acknowledgement

Some code snippets are borrowed from [NeuS](https://github.com/Totoro97/NeuS.git) and [TransparentShapeRealData](https://github.com/yuyingyeh/TransparentShapeRealData.git). Thanks for these great projects.
