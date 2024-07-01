# TSHRNet - MindSpore

### **MindSpore** implement of

Towards High-Quality Specular Highlight Removal by Leveraging Large-scale Synthetic Data

Gang Fu, Qing Zhang, Lei Zhu, Chunxia Xiao, and Ping Li

In ICCV's 23

[Paper](https://arxiv.org/pdf/2309.06302.pdf)

In this paper, our goal is to remove specular highlight removal for object-level images. In this paper, we propose a three-stage network for specular highlight removal, consisting
of (i) physics-based specular highlight removal, (ii) specular-free refinement, and (iii) tone correction. In addition, we present a large-scale synthetic dataset of object-level
images, in which each input image has corresponding albedo, shading, specular residue, diffuse, and tone-corrected diffuse images.

## Prerequisities of MindSpore implementation

```
mindspore = 2.1.1
matplotlib
pillow
tqdm
```

## Datasets

* Our SHHR dataset is available at [OneDrive](https://polyuit-my.sharepoint.com/:u:/g/personal/gangfu_polyu_edu_hk/ERVx4DV78jxGq-1HCPmRsssBOYHPvL_eYmKbGMrELxm8uw?e=tdDAeu)
  or [Google Drive](https://drive.google.com/file/d/1iBBYIvF5ujLuUe6l22eArFRxFPYAPLVR/view?usp=sharing) (~5G).

## Pretrained models

* Pretrained **MindSpore** models on SSHR are available at [checkpoints_SSHR_ms](https://drive.google.com/drive/folders/1mWO1qw8WfdLYqlGfj4Yxuz5vQz3qkOSp?usp=drive_link).
* Also available at [release](https://github.com/nauyihsnehs/TSHRNet-MindSpore/releases).

## Training

```
python train.py \
       -trdd ${train_data_dir} \
       -trdlf ${train_data_list_file} \
       -dn ${dataset_name}
```

## Testing

Note thatwe split "test.lst" into four parts for testin, due to out of memory.

```
python test.py \
       -mn ${model_name} \
       -l ${num_checkpoint} \
       -tdn ${testing_data_name} \
       -tedd ${testing_data_dir} \
       -tedlf ${testing_data_list_file}
```

## Citation

```
@inproceedings{fu-2023-towar-a,
  author =     {Fu, Gang and Zhang, Qing and Zhu, Lei and Xiao, Chunxia and Li, Ping},
  title =     {Towards high-quality specular highlight removal by leveraging large-scale synthetic data},
  booktitle =     {Proceedings of the IEEE International Conference on Computer Vision},
  year =     {2023},
  pages =     {To appear},
}
```
