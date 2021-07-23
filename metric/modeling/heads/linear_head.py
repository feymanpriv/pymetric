# encoding: utf-8
"""
    based on 
    https://github.com/JDAI-CV/fast-reid/blob/master/fastreid/modeling/heads/linear_head.py
"""
import torch
from torch import nn

import metric.core.net as net
from metric.core.config import cfg
from metric.modeling.layers import Arcface, Circle
from metric.modeling.layers import GeneralizedMeanPoolingP, AdaptiveAvgMaxPool2d, FastGlobalAvgPool2d


class LinearHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool_type = cfg.MODEL.HEADS.POOL_LAYER
        if self.pool_type == 'avgpool':      
            self.pool_layer = FastGlobalAvgPool2d()
        elif self.pool_type == 'maxpool':    
            self.pool_layer = nn.AdaptiveMaxPool2d(1)
        elif self.pool_type == 'gempool':    
            self.pool_layer = GeneralizedMeanPoolingP()
        elif self.pool_type == "avgmaxpool": 
            self.pool_layer = AdaptiveAvgMaxPool2d()
        elif self.pool_type == "identity":   
            self.pool_layer = nn.Identity()
        else:
            raise KeyError(f"{pool_type} is invalid")
        
        self.in_feat = cfg.MODEL.HEADS.IN_FEAT
        self.num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        
        # embedding layer
        self.embedding_size = cfg.MODEL.HEADS.REDUCTION_DIM
        self.embedding_layer = nn.Linear(self.in_feat, self.embedding_size, bias=False)
        self.embedding_layer.apply(net.init_weights_classifier)
        # identity classification layer
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        if cls_type == 'linear':    self.classifier = nn.Linear(self.in_feat, self.num_classes, bias=False)
        elif cls_type == 'arcface': self.classifier = Arcface(self.embedding_size, self.num_classes)
        elif cls_type == 'circle':  self.classifier = Circle(self.embedding_size, self.num_classes)
        else:
            raise KeyError(f"{cls_type} is invalid, please choose from "
                           f"'linear', 'arcface' and 'circle'.")

        self.classifier.apply(net.init_weights_classifier)

    def forward(self, features, targets=None):
        global_feat = self.pool_layer(features)
        if self.pool_type != "identity":
            global_feat = global_feat[..., 0, 0]
        global_feat = self.embedding_layer(global_feat)
        #if not self.training: return global_feat
        # training
        try:              pred_class_logits = self.classifier(global_feat)
        except TypeError: pred_class_logits = self.classifier(global_feat, targets)
        return pred_class_logits, global_feat, targets

