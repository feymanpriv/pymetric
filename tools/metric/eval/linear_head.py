# encoding: utf-8
"""
    for feature extraction
"""

import torch
from torch import nn

import metric.core.net as net
from metric.core.config import cfg
from metric.modeling.layers import GeneralizedMeanPoolingP, AdaptiveAvgMaxPool2d, FastGlobalAvgPool2d


class LinearHead(nn.Module):
    def __init__(self):
        super().__init__()

        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        if pool_type == 'avgpool':      
            self.pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'maxpool':    
            self.pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempool':    
            self.pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == "avgmaxpool": 
            self.pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == "identity":   
            self.pool_layer = nn.Identity()
        else:
            raise KeyError(f"{pool_type} is invalid")
        
        self.in_feat = cfg.MODEL.HEADS.IN_FEAT
        self.num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        
        # embedding layer
        self.embedding_size = cfg.MODEL.HEADS.REDUCTION_DIM
        self.embedding_layer = nn.Linear(self.in_feat, self.embedding_size, bias=False)
        self.embedding_layer.apply(net.init_weights_classifier)

    def forward(self, features):
        global_feat = self.pool_layer(features)
        global_feat = global_feat[..., 0, 0]
        global_feat = self.embedding_layer(global_feat)
        return global_feat
    
