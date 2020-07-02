# encoding: utf-8
"""
    based on 
    https://github.com/JDAI-CV/fast-reid/blob/master/fastreid/modeling/heads/linear_head.py
"""

from metric.core.config import cfg
from metric.modeling.layers import *
import metric.core.net as net


class LinearHead(nn.Module):
    def __init__(self, cfg, in_feat, num_classes, pool_layer):
        super().__init__()
        self.pool_layer = pool_layer

        # identity classification layer
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        if cls_type == 'linear':    self.classifier = nn.Linear(in_feat, num_classes, bias=False)
        elif cls_type == 'arcface': self.classifier = Arcface(cfg, in_feat, num_classes)
        elif cls_type == 'circle':  self.classifier = Circle(cfg, in_feat, num_classes)
        else:
            raise KeyError(f"{cls_type} is invalid, please choose from "
                           f"'linear', 'arcface' and 'circle'.")

        self.classifier.apply(net.init_weights_classifier)

    def forward(self, features, targets=None):
        global_feat = self.pool_layer(features)
        global_feat = global_feat[..., 0, 0]
        if not self.training: return global_feat
        # training
        try:              pred_class_logits = self.classifier(global_feat)
        except TypeError: pred_class_logits = self.classifier(global_feat, targets)
        return pred_class_logits, global_feat, targets

