"""
Code source: https://github.com/pytorch/vision
"""

from __future__ import division, absolute_import
import torch.utils.model_zoo as model_zoo
from torch import nn
import torchvision
from torchvision.models.resnet import resnext50_32x4d,resnet50,resnet101,resnet152,Bottleneck
import random
import torch
import math
import os.path as osp
import copy
import numpy as np
from torch.nn import functional as F
# from torch_multi_head_attention import MultiHeadAttention
from .decoder import Decoder
from .GD import Generator,weights_init
from .opts import get_opts, Imagenet_mean, Imagenet_stddev

__all__ = ['bfe']

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)



class ChannelAttn(nn.Module):
    """Channel Attention (Sec. 3.1.I.2)"""

    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttn, self).__init__()
        assert in_channels % reduction_rate == 0
        self.conv1 = ConvBlock(in_channels, in_channels // reduction_rate, 1)
        self.conv2 = ConvBlock(in_channels // reduction_rate, in_channels, 1)

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:])
        # excitation operation (2 conv layers)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ConvBlock(nn.Module):
    """Basic convolutional block.
    
    convolution + batch normalization + relu.
    Args:
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
    """

    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class SoftAttn(nn.Module):
    """Soft Attention (Sec. 3.1.I)
    
    Aim: Spatial Attention + Channel Attention
    
    Output: attention maps with shape identical to input.
    """

    def __init__(self, in_channels):
        super(SoftAttn, self).__init__()
        self.spatial_attn = SpatialAttn()
        self.channel_attn = ChannelAttn(in_channels)
        self.conv = ConvBlock(in_channels, in_channels, 1)

    def forward(self, x):
        y_spatial = self.spatial_attn(x)
        y_channel = self.channel_attn(x)
        y = y_spatial * y_channel
        y = torch.sigmoid(self.conv(y))
        return y


class HardAttn(nn.Module):
    """Hard Attention (Sec. 3.1.II)"""

    def __init__(self, in_channels):
        super(HardAttn, self).__init__()
        self.fc = nn.Linear(in_channels, 4 * 2)
        self.init_params()

    def init_params(self):
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(
            torch.tensor(
                [0, -0.75, 0, -0.25, 0, 0.25, 0, 0.75], dtype=torch.float
            )
        )

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), x.size(1))
        # predict transformation parameters
        theta = torch.tanh(self.fc(x))
        theta = theta.view(-1, 4, 2)
        return theta

class SpatialAttn(nn.Module):
    """Spatial Attention (Sec. 3.1.I.1)"""

    def __init__(self):
        super(SpatialAttn, self).__init__()
        self.conv1 = ConvBlock(1, 1, 3, s=2, p=1)
        self.conv2 = ConvBlock(1, 1, 1)

    def forward(self, x):
        # global cross-channel averaging
        x = x.mean(1, keepdim=True)
        # 3-by-3 conv
        x = self.conv1(x)
        # bilinear resizing
        x = F.upsample(
            x, (x.size(2) * 2, x.size(3) * 2),
            mode='bilinear',
            align_corners=True
        )
        # scaling conv
        x = self.conv2(x)
        return x
class HarmAttn(nn.Module):
    """Harmonious Attention (Sec. 3.1)"""

    def __init__(self, in_channels):
        super(HarmAttn, self).__init__()
        self.soft_attn = SoftAttn(in_channels)
        self.hard_attn = HardAttn(in_channels)

    def forward(self, x):
        y_soft_attn = self.soft_attn(x)
        return y_soft_attn

class FC(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(FC, self).__init__()
        self.fc = nn.Linear(inplanes, outplanes)
        self.bn = nn.BatchNorm1d(outplanes)
        self.act = nn.PReLU()

    def forward(self, x):
        x = self.fc(x)
        return self.act(x)

import cv2
import random
class ResNet(nn.Module):
    def __init__(self, num_classes, fc_dims=None, loss=None, dropout_p=None,  **kwargs):
        super(ResNet, self).__init__()
        resnet_ = resnet50(pretrained=True)
        # resnet_ = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        self.loss = loss
        self.layer0 = nn.Sequential(
            resnet_.conv1,
            resnet_.bn1,
            resnet_.relu,
            resnet_.maxpool)
        self.layer1 = resnet_.layer1
        self.layer2 = resnet_.layer2
        self.layer3 = resnet_.layer3
        # self.att1 = CBAM_Module(1024)
        # self.att2 = CBAM_Module(1024)
        # self.att_ori = CBAM_Module(1024)

        layer4 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        layer4.load_state_dict(resnet_.layer4.state_dict())

        self.layer41 = nn.Sequential(copy.deepcopy(layer4))
        self.layer42 = nn.Sequential(copy.deepcopy(layer4))
        self.layer43 = nn.Sequential(copy.deepcopy(layer4))
        self.layer44 = nn.Sequential(copy.deepcopy(layer4))

        self.res_part1 = Bottleneck(2048, 512)
        self.res_part2 = Bottleneck(2048, 512)
        self.res_part3 = Bottleneck(2048, 512)

        self.reduction1 = nn.Sequential(
            nn.Linear(2048, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )
        self.reduction1.apply(weights_init_kaiming)

        self.reduction2 = nn.Sequential(
            nn.Linear(2048, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )
        self.reduction2.apply(weights_init_kaiming)

        self.reduction3 = nn.Sequential(
            nn.Linear(2048, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )
        self.reduction3.apply(weights_init_kaiming)

        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.harm1=HarmAttn(512)
        self.harm2=HarmAttn(1024)

        self.classifier1 = nn.Linear(2048, num_classes)
        self.classifier2 = nn.Linear(2048, num_classes)
        self.classifier4 = nn.Linear(2048*3, num_classes)
        self.classifier3 = nn.Linear(2048, num_classes)
        nn.init.normal_(self.classifier1.weight, 0, 0.01)
        if self.classifier1.bias is not None:
            nn.init.constant_(self.classifier1.bias, 0)
        nn.init.normal_(self.classifier2.weight, 0, 0.01)
        if self.classifier2.bias is not None:
            nn.init.constant_(self.classifier2.bias, 0)
        nn.init.normal_(self.classifier3.weight, 0, 0.01)
        if self.classifier3.bias is not None:
            nn.init.constant_(self.classifier3.bias, 0)
        nn.init.normal_(self.classifier4.weight, 0, 0.01)
        if self.classifier4.bias is not None:
            nn.init.constant_(self.classifier4.bias, 0)

        self.numid=0        
        
    def featuremaps_my(self, x, x_2, x_3):
        if self.training:
            b = x.size(0)
            x = torch.cat([x, x_2, x_3], 0)
           
        x = self.layer0(x)   # 64, 96, 32
        x = self.layer1(x)   # 256, 96, 32
        x = self.layer2(x)   # 512, 48, 16
        x_atten1 = self.harm1(x)
        x = x*x_atten1
        x = self.layer3(x)   # 1024, 24, 8
        x_atten2= self.harm2(x)
        x = x*x_atten2
        if self.training:
            x_1 = x[:b, :, :, :]
            x_2 = x[b:2*b, :, :, :]
            x_3 = x[2*b:3*b, :, :, :]

            x_1 = self.layer41(x_1)  
            x_2 = self.layer42(x_2)
            x_3 = self.layer43(x_3)

        else:
            x_1 = self.layer41(x)
            x_2 = self.layer42(x)
            x_3 = self.layer43(x)

        return x_1, x_2, x_3
     
    def featuremaps_my_train(self, x):
        x = self.layer0(x)   # 64, 96, 32
        x = self.layer1(x)   # 256, 96, 32
        x = self.layer2(x)   # 512, 48, 16
        x_atten1 = self.harm1(x)
        x = x*x_atten1
        x = self.layer3(x)   # 1024, 24, 8
        x_atten2= self.harm2(x)
        x = x*x_atten2
        x_1 = self.layer41(x)
        x_2 = self.layer42(x)
        x_3 = self.layer43(x)

        return x_1, x_2, x_3


    def featuremaps(self, x):
        x = self.layer0(x)   # 64, 96, 32
        x = self.layer1(x)   # 256, 96, 32
        x = self.layer2(x)   # 512, 48, 16
        x_atten1 = self.harm1(x)
        x = x*x_atten1
        x = self.layer3(x)   # 1024, 24, 8
        x_atten2= self.harm2(x)
        x = x*x_atten2
        x_1 = self.layer41(x)
        x_2 = self.layer42(x)
        x_3 = self.layer43(x)
        return (x_1 + x_2 + x_3) / 3

    def forward(self, x, x_2=None, x_3=None, return_featuremaps=False, state='test', segmentmask=False,epoch=None,returnflag=False, update_param = False):
        if update_param == True:
            self.update_param()
            return
        if self.training:
            b = x.size(0)
            if return_featuremaps:
                features = self.featuremaps(x)
                return features.detach()

            f1, f2, f3 = self.featuremaps_my(x, x_2, x_3)

            f1 = self.res_part1(f1)
            f2 = self.res_part2(f2)
            f3 = self.res_part3(f3)

            v1 = self.global_maxpool(f1)
            v1 = v1.view(v1.size(0), -1)
            v1_1 = self.reduction1(v1)  # 512

            v2 = self.global_maxpool(f2)
            v2 = v2.view(v2.size(0), -1)
            v2_1 = self.reduction2(v2)

            v3 = self.global_maxpool(f3)
            v3 = v3.view(v3.size(0), -1)
            v3_1 = self.reduction3(v3)  # 512
            
            v4_1 = torch.cat([v1_1, v2_1, v3_1], dim = 1)

            y1 = self.classifier1(v1_1)
            y2 = self.classifier2(v2_1)
            y3 = self.classifier3(v3_1)
            y4 = self.classifier4(v4_1)

            if self.loss == 'softmax':
                return y1, y2, y3, y4
            # return y1, y2, y3, y4, mask_1
            elif self.loss == 'triplet':
                return y1, y2, y3, y4
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))

        if not self.training:
            if return_featuremaps:
                feature = self.featuremaps(x)
                return feature

            f1, f2, f3= self.featuremaps_my(x, x, x)



            f1 = self.res_part1(f1)

            f2 = self.res_part2(f2)

            f3 = self.res_part3(f3)

            if return_featuremaps:
                return f1

            v1 = self.global_maxpool(f1)
            v1 = v1.view(v1.size(0), -1)
            v1_1 = self.reduction1(v1)  # 512

            v2 = self.global_maxpool(f2)
            v2 = v2.view(v2.size(0), -1)
            v2_1 = self.reduction2(v2)

            v3 = self.global_maxpool(f3)
            v3 = v3.view(v3.size(0), -1)
            v3_1 = self.reduction3(v3)  # 512

            v1_1 = F.normalize(v1_1, p=2, dim=1)
            v2_1 = F.normalize(v2_1, p=2, dim=1)
            v3_1 = F.normalize(v3_1, p=2, dim=1)

            return torch.cat([v1_1, v2_1, v3_1], 1)


# ResNet
def bfe(num_classes, loss='softmax', pretrained=True, **kwargs):
    G = Generator(3, 1, 16, 'bn').apply(weights_init)
    model = ResNet(
        num_classes=num_classes,
        fc_dims=None,
        loss=loss,
        dropout_p=None,
        **kwargs
    )
    return G, model

