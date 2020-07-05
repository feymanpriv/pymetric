#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Train a metric model."""

import metric.core.config as config
import metric.core.distributed as dist
import metric.core.trainer as trainer
from metric.core.config import cfg


def main():
    config.load_cfg_fom_args("Train a classification model.")
    config.assert_and_infer_cfg()
    cfg.freeze()
    dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.train_model)


if __name__ == "__main__":
    main()
