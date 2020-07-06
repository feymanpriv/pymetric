# pymetric

**pymetric** tries to build a metric learning codebase based on [PyTorch](https://pytorch.org/). It refers from **pycls** [On Network Design Spaces for Visual Recognition](https://arxiv.org/abs/1905.13214) project and **fast-reid** from (https://github.com/JDAI-CV/fast-reid).

# Introduction

- Includes **pycls** (https://github.com/facebookresearch/pycls), refer to [`pycls.md`](docs/pycls.md).
- **pymetric** is written by `DistributedDataParallel` which is different from **fast-reid**. Now mainly includes features such as arcface loss and circle loss, ongoing.

# Installation

**Requirements:**

- NVIDIA GPU, Linux, Python3(tested on 3.6.10)
- PyTorch, various Python packages; Instructions for installing these dependencies are found below

**Notes:**

- **pymetric** does not currently support running on CPU; a GPU system is required
- **pymetric** has been tested with CUDA 10.2 and cuDNN 7.1

## PyTorch

To install PyTorch with CUDA support, follow the [installation instructions](https://pytorch.org/get-started/locally/) from the [PyTorch website](https://pytorch.org).

## pymetric

Clone the **pymetric** repository:

```
# PYMETRIC=/path/to/clone/pycls
git clone https://github.com/ym547559398/pymetric $PYMETRIC
```

Install Python dependencies:

```
pip install -r $PYMETRIC/requirements.txt
```

Set PYTHONPATH:

```
cd $PYMETRIC && export PYTHONPATH=`pwd`:$PYTHONPATH
```

## Datasets

Same with **pycls**, **pymetric** finds datasets via symlinks from `metric/datasets/data` to the actual locations where the dataset images and annotations are stored.

Expected datasets structure for ImageNet:

```
imagenet
|_ train
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ val
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ ...
```

Create a directory containing symlinks:

```
mkdir -p /path/pycls/pycls/datasets/data
```

Symlink ImageNet:

```
ln -s /path/imagenet /path/pycls/pycls/datasets/data/imagenet
```

Annotation format shows as /path/imagenet/val_list.txt

# Getting Started

Training a metric model:

```
python tools/train_metric.py \
    --cfg configs/metric/R-50-1x64d_step_8gpu.yaml \
    OUT_DIR ./output
```

Training a classfication model:

```
python tools/train_net.py \
    --cfg configs/cls/R-50-1x64d_step_8gpu.yaml \
    OUT_DIR ./output
```

# License

**pymetric** is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
