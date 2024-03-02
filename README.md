# UNISURF: Unifying Neural Implicit Surfaces and Radiance Fields for Multi-View Reconstruction
## [Project Page](https://moechsle.github.io/unisurf/) | [Paper](http://www.cvlibs.net/publications/Oechsle2021ICCV.pdf) | [Supplementary](http://www.cvlibs.net/publications/Oechsle2021ICCV_supplementary.pdf) | [Video](https://www.youtube.com/watch?v=WXUfHvZge0E)

![](./media/unisurf_teaser.gif)

This repository contains the implementation of the paper:

UNISURF: Unifying Neural Implicit Surfaces and Radiance Fields for Multi-View Reconstruction  
[Michael Oechsle](https://moechsle.github.io/), [Songyou Peng](https://pengsongyou.github.io/), and [Andreas Geiger](http://www.cvlibs.net/)  
**ICCV 2021 (Oral)**  

Neural implicit 3D representations have emerged as a powerful paradigm for reconstructing surfaces from multi-view images and synthesizing novel views. Unfortunately, existing methods such as DVR or IDR require accurate per-pixel object masks as supervision. At the same time, neural radiance fields have revolutionized novel view synthesis. However, NeRF's estimated volume density does not admit accurate surface reconstruction. Our key insight is that implicit surface models and radiance fields can be formulated in a unified way, enabling both surface and volume rendering using the same model. This unified perspective enables novel, more efficient sampling procedures and the ability to reconstruct accurate surfaces without input masks. We compare our method on the DTU, BlendedMVS, and a synthetic indoor dataset. Our experiments demonstrate that we outperform NeRF in terms of reconstruction quality while performing on par with IDR without requiring masks.

If you find our code or paper useful, please cite as
```bibtex
@inproceedings{Oechsle2021ICCV,
  author    = {Michael Oechsle and Songyou Peng and Andreas Geiger},
  title     = {UNISURF: Unifying Neural Implicit Surfaces and Radiance Fields for Multi-View Reconstruction},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year      = {2021}
} 
```    
    
## Installation

First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/).

You can create an anaconda environment called `unisurf` using
```
conda env create -f environment.yaml
conda activate unisurf
```
Next, compile the extension modules.
You can do this via
```
python setup.py build_ext --inplace
```

## Dataset
For downloading the preprocessed data, run the following script. The data is adapted from [DVR](https://github.com/autonomousvision/differentiable_volumetric_rendering) and [IDR](https://github.com/lioryariv/idr)
```
source ./download_dataset.sh
```

## Extract mesh from a pretrained model

If you just want to quickly extract a mesh from a pre-trained model, you can run our demo with 
```
python extract_mesh.py configs/DTU_pre/scan0$id.yaml 
```

You can find predicted meshes in `out/scan_0$id/`.


## Training
For training a model from scratch run 
```
python train.py configs/DTU/scan_0$id.yaml
```
in the conda environment.
You can use tensor board to visualize the training process. The logfiles are saved to the `logs` folder in the output directory.
```
tensorboard --logdir ./out --port 6006
```

