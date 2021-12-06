# pymetric

**pymetric** tries to build a metric learning and retrieval codebase based on [PyTorch](https://pytorch.org/). It refers from **pycls** [On Network Design Spaces for Visual Recognition](https://arxiv.org/abs/1905.13214) project and **fast-reid** from (https://github.com/JDAI-CV/fast-reid).

# Introduction

- **2nd** place in Google Landmark Retrieval 2020 Competition. (https://drive.google.com/file/d/1XnzxMOHhzua9tjrAjo-X55ieKVJzRJw_/view?usp=sharing)
- Includes **pycls** (https://github.com/facebookresearch/pycls), refer to [`pycls.md`](docs/pycls.md).
- **pymetric** is written by `DistributedDataParallel` which is different from **fast-reid**. Now mainly includes features such as arcface loss and circle loss, ongoing.
- Add multicards feature extraction, searching topk and computing mAP.

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
# PYMETRIC=/path/to/clone/pymetric
git clone https://github.com/feymanpriv/pymetric $PYMETRIC
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

Same with **pycls**, **pymetric** finds datasets via symlinks from `metric/datasets/data` to the actual locations where the dataset images and annotations are stored. Refer to [`DATA.md`](docs/DATA.md).


# Getting Started

Training a metric model:

```
python tools/train_metric.py \
    --cfg configs/metric/R-50-1x64d_step_8gpu.yaml \
    OUT_DIR ./output \
    PORT 12001 \
    TRAIN.WEIGHTS path/to/pretrainedmodel
```
Resume training: 

```
python tools/train_metric.py \
    --cfg configs/metric/R-50-1x64d_step_8gpu.yaml \
    OUT_DIR ./output \
    PORT 12001 \
    TRAIN.AUTO_RESUME True
```

Extracting features(labels) and evaluation
```
set ${total_num} = n*(gpu_cards)
sh tools/metric/eval/infer.sh
python search.py search_gpu ${queryfea_path}, ${referfea_path}, ${output}

Convert to tensorflow2.3 (please refer onnx and onnx-tensorflow)
examples: tools/convert/torch2onnx.py tools/convert/onnx2tf.py
```

## Pretrained weights
-[resnet50](https://pan.baidu.com/s/1WAGiz5EHJrKT-61m-B322Q) (c3l3)

-[resnet101](https://pan.baidu.com/s/1uzh6_Si-6ZCsoxS1Au8MZQ) (t3ln)

-[resnest269](https://pan.baidu.com/s/1Hf6C9qSuH_qllfizwx5QeA) (3c5a)


# Results

**2nd place on Google Landmark Retrieval Challenge 2020**

|  Backbone    |  Scale  | Margin |   Size  | Public Score | Private Score |
|--------------|:-------:|:------:|:-------:|:------------:|:-------------:|
|  ResNeSt269  |    30   |  0.15  | 224/224 |   0.35129    |    0.30819    |
|  ResNeSt269  |    30   |  0.15  | 448/448 |   0.36972    |    0.33015    |
|  ResNeSt269  |    30   |  0.15  | 640/448 |   0.39040    |    0.34718    |


# License

**pymetric** is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
