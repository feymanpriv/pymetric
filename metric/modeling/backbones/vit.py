#!/usr/bin/env python3
# Copyright (c) OpenAI, Inc. and its affiliates.
# CLIP (ViT)
# written by feymanpriv

""" Vision Transformer models """

from collections import OrderedDict
import metric.core.net as net
from metric.core.config import cfg

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualAttentionBlock(nn.Module):
    """ Vision Transformer Block """

    def __init__(self, d_model, n_head, attn_mask):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model))
                    ]))        
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
                            if self.attn_mask is not None else None 
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        
    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """ Transformer Arch """

    def __init__(self, width, layers, heads, attn_mask=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) 
                                        for _ in range(layers)])
        
    def forward(self, x):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    """ Vision Transformer Network """
    
    def __init__(self):
        super().__init__()
        self._construct() 
        self.apply(net.init_weights)
        self.init_weights()

    def _construct(self):
        input_resolution, output_dim = int(cfg.TRAIN.IM_SIZE), int(cfg.MODEL.EMB_DIM)
        width, layers = int(cfg.MODEL.WIDTH), int(cfg.MODEL.LAYERS)
        patch_size, heads = int(cfg.MODEL.PATCH_SIZE), width // 64
        scale = width ** -0.5

        self.conv1 = nn.Conv2d(3, width, patch_size, stride=patch_size, bias=False)
        self.class_embedding = nn.Parameter(scale * torch.randn(width))    
        self.positional_embedding = nn.Parameter(scale * torch.randn(
                                    (input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim = 1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj            
        return x

    def init_weights(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
         

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """ GELU Reinplementation """

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

