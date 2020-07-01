# Getting Started

This document provides basic instructions for training and evaluation using **pycls**.

- For general information about **pycls**, please see [`README.md`](../README.md)
- For installation instructions, please see [`INSTALL.md`](INSTALL.md)

## Training Models

Training on ImageNet with 8 GPUs:

```
python tools/train_net.py \
    --cfg configs/R-50-1x64d_step_8gpu.yaml \
    OUT_DIR ./output
```

## Finetuning Models

Finetuning on ImageNet with 1 GPU:

```
python tools/train_net.py \
    --cfg configs/archive/imagenet/resnet/R-50-1x64d_step_1gpu.yaml \
    TRAIN.WEIGHTS /path/to/weights/file \
    OUT_DIR /tmp
```

## Evaluating Models

Evaluation on ImageNet with 1 GPU:

```
python tools/test_net.py \
    --cfg configs/archive/imagenet/resnet/R-50-1x64d_step_1gpu.yaml \
    TEST.WEIGHTS /path/to/weights/file \
    OUT_DIR /tmp
```
