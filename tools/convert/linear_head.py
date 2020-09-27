# encoding: utf-8
"""
    based on 
    https://github.com/JDAI-CV/fast-reid/blob/master/fastreid/modeling/heads/linear_head.py
"""
import torch
from torch import nn

import metric.core.net as net
from metric.core.config import cfg
from metric.modeling.layers import Arcface, Circle, ArcMarginProduct
from metric.modeling.layers import GeneralizedMeanPoolingP, AdaptiveAvgMaxPool2d, FastGlobalAvgPool2d, RMAC


class LinearHead(nn.Module):
    def __init__(self):
        super().__init__()

        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        if pool_type == 'avgpool':      
            self.pool_layer = FastGlobalAvgPool2d()
            #self.pool_layer = nn.AdaptiveAvgPool2d((1, 1)) 
        elif pool_type == 'maxpool':    
            self.pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempool':    
            self.pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == "avgmaxpool": 
            self.pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == "identity":   
            self.pool_layer = nn.Identity()
        elif pool_type == "combine":
            self.pool_layer = RMAC()
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

class LinearHead_cat_with_pred(nn.Module):
    def __init__(self):
        super().__init__()

        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        if pool_type == 'avgpool':      
            self.pool_layer = FastGlobalAvgPool2d()
            #self.pool_layer = nn.AdaptiveAvgPool2d((1, 1)) 
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

        print("num_classes", self.num_classes)
        print("embedding_size", self.embedding_size)
        #cls predict layer
        #self.classifier = Arcface(self.embedding_size, self.num_classes)
        self.classifier = ArcMarginProduct(self.embedding_size, self.num_classes)

    

    def forward(self, features, labels=torch.ones([1]).long()):
        global_feat = self.pool_layer(features)
        global_feat = global_feat[..., 0, 0]
        global_feat = self.embedding_layer(global_feat)
        pred_class_logits = self.classifier(global_feat,labels)
        #
        pred_class = nn.functional.softmax(pred_class_logits,dim=1)
        """
        max_ids = torch.argmax(pred_class, dim=1)
        pred_score_list = []
        for i, id in enumerate(max_ids):
            pred_score_list.append(pred_class[i,id])
        max_ids = max_ids.float().unsqueeze(1)
        pred_scores = torch.tensor(pred_score_list).float().unsqueeze(1)
        global_feat = torch.cat([global_feat, max_ids, pred_scores], axis=1) 
        print(global_feat.shape)
        """
        #pred_scores, max_ids =  torch.max(pred_class, dim=1)
        
        #pred_scores = pred_scores.float().unsqueeze(1)#.cuda()
        #max_ids = max_ids.float().unsqueeze(1)#.cuda()
        
        #global_feat = torch.cat([global_feat, max_ids, pred_scores], axis=1)
        #print(pred_scores.shape, max_ids.shape)
        #global_feat = torch.cat([global_feat, pred_class], axis=1)
        #print(global_feat.shape) 
        #print(pred_class.shape)
        print(pred_class_logits.shape)
        return pred_class_logits 
