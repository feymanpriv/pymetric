# Installation Instructions

This document covers how to install **pycls** and its dependencies.

- For general information about **pycls**, please see [`README.md`](../README.md)

**Requirements:**

- NVIDIA GPU, Linux, Python3(tested on 3.6.10)
- PyTorch, various Python packages; Instructions for installing these dependencies are found below

**Notes:**

- **pycls** does not currently support running on CPU; a GPU system is required
- **pycls** has been tested with CUDA 10.2 and cuDNN 7.1

## PyTorch

To install PyTorch with CUDA support, follow the [installation instructions](https://pytorch.org/get-started/locally/) from the [PyTorch website](https://pytorch.org).

## pycls

Clone the **pycls** repository:

```
# PYCLS=/path/to/clone/pycls
git clone https://github.com/facebookresearch/pycls $PYCLS
```

Install Python dependencies:

```
pip install -r $PYCLS/requirements.txt
```

Set up Python modules:

```
cd $PYCLS && python setup.py develop --user
```

## Datasets

**pycls** finds datasets via symlinks from `pycls/datasets/data` to the actual locations where the dataset images and annotations are stored.

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


## Getting Started

Training R-50 on ImageNet with 8 GPUs:

```
python tools/train_net.py \
    --cfg configs/R-50-1x64d_step_8gpu.yaml \
    OUT_DIR ./output
```

Please see [`GETTING_STARTED.md`](GETTING_STARTED.md) for basic instructions on training and evaluation with **pycls**.
