# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from mmseg.models.utils import Upsample
from .DIL_decode_head import DIL_BaseDecodeHead

class GCN(nn.Module):
    def __init__(self, dim_1_channels, dim_2_channels):
        super().__init__()
        self.conv1d_1 = nn.Conv1d(dim_1_channels, dim_1_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1d_2 = nn.Conv1d(dim_2_channels, dim_2_channels, 1)

    def forward(self, x):
        h = self.conv1d_1(x).permute(0, 2, 1)
        h=h+x.permute(0, 2, 1)
        return self.conv1d_2(self.relu(h)).permute(0, 2, 1)

class SE(nn.Module):
    def __init__(self, channel, reduction):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return y

class SAM(nn.Module):
    def __init__(self,kernel_size=7):
        super(SAM, self).__init__()
        self.conv=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=kernel_size,stride=1,padding=kernel_size//2,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out,_=torch.max(x,dim=1,keepdim=True)
        mean_out=torch.mean(x,dim=1,keepdim=True)
        out=torch.cat((max_out,mean_out),dim=1)
        out=self.sigmoid(self.conv(out))
        return out

@MODELS.register_module()
class GSMF_RS_DIL_Head(DIL_BaseDecodeHead):
    def __init__(self, mid_channels=256, N=64, reduction=8, kernel_size=7, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.mid_channels = mid_channels
        self.N = N
        self.reduction = reduction
        self.kernel_size = kernel_size
        self.lateral_convs = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.cam = nn.ModuleList()
        self.sam = nn.ModuleList()
        self.bottleneck = ConvModule(
            self.in_channels[-1],
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.phi = ConvModule(
            self.channels,
            self.mid_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.theta = ConvModule(
            self.channels,
            self.N,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.gcn = GCN(self.N, self.mid_channels)
        self.blocker = nn.BatchNorm2d(self.channels, eps=1e-04)
        self.phi_inv = ConvModule(
            self.mid_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        for i in range(1,len(self.in_index)):
            self.lateral_convs.append(ConvModule(
                self.in_channels[len(self.in_index)-i-1],
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False))
            self.upsample.append(
                Upsample(
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=self.align_corners))
            self.cam.append(SE(self.channels, self.reduction))
            self.sam.append(SAM(kernel_size=self.kernel_size))

    def forward(self, inputs):
        fpn_out = self.bottleneck(inputs[-1])

        b, C, H, W = fpn_out.size()

        B = self.theta(fpn_out).view(b, self.N, -1)
        x_reduced = self.phi(fpn_out).view(b, self.mid_channels, H * W)
        x_reduced = x_reduced.permute(0, 2, 1)
        v = B.bmm(x_reduced)

        z = self.gcn(v)
        y = B.permute(0, 2, 1).bmm(z).permute(0, 2, 1)
        y = y.view(b, self.mid_channels, H, W)
        x_res = self.blocker(self.phi_inv(y))

        lateral_output = fpn_out + x_res

        for i in range(1, len(self.in_index)):
            lateral_output_up = self.upsample[i - 1](lateral_output)
            c_attention = self.cam[i-1](lateral_output_up)
            lateral_output = self.lateral_convs[i - 1](inputs[len(self.in_index) - 1 - i])
            s_attention = self.sam[i - 1](lateral_output)
            lateral_output_ = lateral_output * c_attention
            lateral_output_up_ = lateral_output_up * s_attention
            lateral_output = lateral_output_up_ + lateral_output_ + lateral_output_up + lateral_output
        output = self.cls_seg(lateral_output)
        return output
