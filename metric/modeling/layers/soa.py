import torch
import torch.nn as nn
import metric.core.net as net
from metric.core.config import cfg


class SOABlock(nn.Module):
    def __init__(self, in_ch):
        super(SOABlock, self).__init__()

        self.in_ch = in_ch
        self.out_ch = in_ch
        self.mid_ch = in_ch // 2

        print('Num channels:  in    out    mid')
        print('               {:>4d}  {:>4d}  {:>4d}'.format(self.in_ch, self.out_ch, self.mid_ch))

        self.f = nn.Sequential(
                nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1)),
                nn.BatchNorm2d(self.mid_ch, eps=cfg.BN.EPS, momentum=cfg.BN.MOM),
                nn.ReLU())
        self.g = nn.Sequential(
                nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1)),
                nn.BatchNorm2d(self.mid_ch, eps=cfg.BN.EPS, momentum=cfg.BN.MOM),
                nn.ReLU())
        self.h = nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1))
        self.v =nn.Conv2d(self.mid_ch, self.out_ch, (1, 1), (1, 1))

        self.softmax = nn.Softmax(dim=-1)

        for conv in [self.f, self.g, self.h]:    #, self.v]:
            conv.apply(net.init_weights)

        self.v.apply(net.init_weights_classifier)


    def forward(self, x):
        B, C, H, W = x.shape

        f_x = self.f(x).view(B, self.mid_ch, H * W) # B * mid_ch * N, where N = H*W
        g_x = self.g(x).view(B, self.mid_ch, H * W) # B * mid_ch * N, where N = H*W
        h_x = self.h(x).view(B, self.mid_ch, H * W) # B * mid_ch * N, where N = H*W

        z = torch.bmm(f_x.permute(0, 2, 1), g_x) # B * N * N, where N = H*W
        attn = self.softmax((self.mid_ch ** -.50) * z)

        z = torch.bmm(attn, h_x.permute(0, 2, 1)) # B * N * mid_ch, where N = H*W
        z = z.permute(0, 2, 1).view(B, self.mid_ch, H, W) # B * mid_ch * H * W

        z = self.v(z)
        z = z + x

        return z
