# pymetric

**pymetric** tries to build a metric learning and retrieval codebase based on [PyTorch](https://pytorch.org/). It refers from **pycls** [On Network Design Spaces for Visual Recognition](https://arxiv.org/abs/1905.13214) project and **fast-reid** from (https://github.com/JDAI-CV/fast-reid).

# Introduction

- 2nd place in Google Landmark Retrieval 2020 Competition
- Includes **pycls** (https://github.com/facebookresearch/pycls), refer to [`pycls.md`](docs/pycls.md).
- **pymetric** is written by `DistributedDataParallel` which is different from **fast-reid**. Now mainly includes features such as arcface loss and circle loss, ongoing.
- Add multicards feature extraction, searching topk and computing mAP

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

Extracting features and evaluation
```
set ${total_num} = n*(gpu_cards)
sh tools/metric/eval/infer.sh
python search.py search_gpu ${queryfea_path}, ${referfea_path}, ${output}
```

# License

**pymetric** is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
