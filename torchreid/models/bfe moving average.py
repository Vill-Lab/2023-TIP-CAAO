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
from torch_multi_head_attention import MultiHeadAttention
from .decoder import Decoder
from .GD import Generator,weights_init
from .opts import get_opts, Imagenet_mean, Imagenet_stddev

__all__ = ['bfe']

class PatchShuffle(nn.Module):
    def __init__(self, patch_size=(2,2), shuffle_probability=1):
        super().__init__()
        self.patch_size = patch_size
        self.shuffle_probability = shuffle_probability

    def forward(self, T: torch.Tensor) -> torch.Tensor:
        if T.shape[0] != 128:
            # case in which we are evaluating the model (PatchShuffle is NOT applied)
            return T
        patch_size = self.patch_size
        shuffle_probability = self.shuffle_probability
        input_n, input_c, input_h, input_w = T.shape

        T = T.reshape(-1, input_h, input_w)

        # for each feature maps, decide whether to patchshuffle or not
        indices_tensor = []    # will be converted into a (input_n*input_c, input_h, input_w) tensor
        for i in range(T.shape[0]):
            # create the indices map
            idx = np.arange(input_h*input_w).reshape(input_h,input_w)

            flick_patchshuffle = np.random.choice( [True,False], p=(self.shuffle_probability, 1-self.shuffle_probability) )
            if flick_patchshuffle:
                w_patches = input_w // patch_size[1]    # n. patches along width
                h_patches = input_h // patch_size[0]    # n. patches along height
                patches_idx = idx.reshape(h_patches,patch_size[0],w_patches,patch_size[1]).swapaxes(1,2).reshape(-1,patch_size[0],patch_size[1])
                for i,patch_idx in enumerate(patches_idx):
                    patches_idx[i] = np.random.permutation(patch_idx.reshape(-1)).reshape(2,2)


                final = []

                for i in range(h_patches):
                    block_row = []
                    for j in range(w_patches):
                        block_row.append(patches_idx[w_patches*i+j])
                    block_row = np.hstack(block_row)
                    final.append(block_row)
                idx = np.vstack(final)
            
            indices_tensor.append(idx)

        indices_tensor = np.array(indices_tensor).reshape(input_n,input_c,input_h,input_w)

        return T.reshape(-1)[indices_tensor.reshape(-1)].reshape(input_n, input_c, input_h, input_w)
    
    def extra_repr(self) -> str:
        return ('patch_size={patch_size}, shuffle_probability={shuffle_probability}')

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

class CAM_Module(nn.Module):
    """ Channel attention module"""
    
    def __init__(self, channels, reduction=16):
        super(CAM_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SAM_Module(nn.Module):
    """ Position attention module"""

    def __init__(self, channels):
        super(SAM_Module, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv_after_concat = nn.Conv2d(1, 1, kernel_size = 3, stride=1, padding = 1)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        module_input = x
        avg = torch.mean(x, 1, True)
        x = self.conv_after_concat(avg)
        # x = self.relu(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x
        return x


class CBAM_Module(nn.Module):

    def __init__(self, channels, reduction=16):
        super(CBAM_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = nn.Conv2d(2, 1, kernel_size = 3, stride=1, padding = 1)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # channel attention
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)
        x = module_input * x
        # spatial attention
        module_input = x
        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        x = torch.cat((avg, mx), 1)
        x = self.conv_after_concat(x)
        spatial_att_map = self.sigmoid_spatial(x)
        x = module_input * spatial_att_map
        return x



class BatchErasing(nn.Module):
    def __init__(self, sl=0.25, sh=0.5, r1=0.5, mean=[0.4914, 0.4822, 0.4465], Threshold=1):
        super(BatchErasing, self).__init__()        
        self.it = 0
        self.Threshold = Threshold
        self.sx = None
        self.sy = None
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def forward(self, x):
        if self.training:
            for attempt in range(10000000000):
                h, w = x.size()[-2:]
                area = h * w
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1)
                rh = int(round(math.sqrt(target_area * aspect_ratio)))
                rw = int(round(math.sqrt(target_area / aspect_ratio)))
                if rw < w and rh < h:
                    if self.it % self.Threshold == 0:
                        self.sx = random.randint(0, h - rh)
                        self.sy = random.randint(0, w - rw)
                    self.it += 1
                    mask = x.new_ones(x.size())
                    mask[:, :, self.sx:self.sx + rh, self.sy:self.sy + rw] = 0
                    x[:, 0, self.sx:self.sx + rh, self.sy:self.sy + rw] = random.uniform(0, 1)
                    x[:, 1, self.sx:self.sx + rh, self.sy:self.sy + rw] = random.uniform(0, 1)
                    x[:, 2, self.sx:self.sx + rh, self.sy:self.sy + rw] = random.uniform(0, 1)
                    return x#, mask, (self.sx, self.sx+rh, self.sy, self.sy+rw)
        if not self.training:
            return x





class RandomErasing_v2(nn.Module):
    def __init__(self, sl=0.25, sh=0.5, r1=0.5, mean=[0.4914, 0.4822, 0.4465]):
        super(RandomErasing_v2, self).__init__()
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    # img 32,3,384,128
    def forward(self, img,erase_x,erase_y):
        mask = torch.ones(img.size(0), 1, 384, 128, dtype=torch.long)
        for i in range(img.size(0)):
            for attempt in range(10000000000):
                area = img.size()[2] * img.size()[3]
                target_area = random.uniform(self.sl, self.sh) * area#area * (0.35 - 0.2 * epoch / 70)#
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img.size()[3] and h < img.size()[2]:
                    x1 = erase_x[i]
                    y1 = erase_y[i]
                    if x1+h>img.size()[2]:
                        x1=img.size()[2]-h
                    if y1+w>img.size()[3]:
                        y1=img.size()[3]-w
                    if img.size()[1] == 3:
                        img[i, 0, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
                        img[i, 1, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
                        img[i, 2, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
                        mask[i, 0, x1:x1 + h, y1:y1 + w] = 0
                        break
        return img,mask



class AttentSwap(nn.Module):
    def __init__(self, sl=0.3, sh=0.5, r1=0.5, mean=[0.4914, 0.4822, 0.4465]):
        super(AttentSwap, self).__init__()
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
    # img 32,3,384,128
    def forward(self, img,erase_x,erase_y,erase_x_min,erase_y_min):
        img_ori=img.clone()
        for i in range(img.size(0)):
            for attempt in range(10000000000):
                area = img.size()[2] * img.size()[3]
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img.size()[3] and h < img.size()[2]:
                    x1=erase_x[i]
                    x2=erase_x_min[i]
                    y1=erase_y[i]
                    y2=erase_y_min[i]

                    if x1+h > img.size()[2]:
                        x1 = img.size()[2]-h

                    if x2+h > img.size()[2]:
                        x2 = img.size()[2]-h

                    if y1+w > img.size()[3]:
                        y1 = img.size()[3]-w

                    if y2+w > img.size()[3]:
                        y2 = img.size()[3]-w

                    img[i, 0, x1:x1 + h, y1:y1 + w] = img_ori[i, 0, x2:x2 + h, y2:y2 + w]
                    img[i, 1, x1:x1 + h, y1:y1 + w] = img_ori[i, 1, x2:x2 + h, y2:y2 + w]
                    img[i, 2, x1:x1 + h, y1:y1 + w] = img_ori[i, 2, x2:x2 + h, y2:y2 + w]

                    img[i, 0, x2:x2 + h, y2:y2 + w] = img_ori[i, 0, x1:x1 + h, y1:y1 + w]
                    img[i, 1, x2:x2 + h, y2:y2 + w] = img_ori[i, 1, x1:x1 + h, y1:y1 + w]
                    img[i, 2, x2:x2 + h, y2:y2 + w] = img_ori[i, 2, x1:x1 + h, y1:y1 + w]


                    break
        return img




class FC(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(FC, self).__init__()
        self.fc = nn.Linear(inplanes, outplanes)
        self.bn = nn.BatchNorm1d(outplanes)
        self.act = nn.PReLU()

    def forward(self, x):
        x = self.fc(x)
        return self.act(x)
class GDN(nn.Module):
    def __init__(self, inplanes, outplanes, intermediate_dim=256):
        super(GDN, self).__init__()
        self.fc1 = FC(inplanes, intermediate_dim)
        self.fc2 = FC(intermediate_dim, outplanes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        intermediate = self.fc1(x)
        out = self.fc2(intermediate)
        # return intermediate, self.softmax(out)
        return intermediate, torch.softmax(out, dim=1)
class MultiHeads(nn.Module):
    def __init__(self, feature_dim=256, groups=4, mode='S', backbone_fc_dim=1024):
        super(MultiHeads, self).__init__()
        self.mode = mode
        self.groups = groups
        # self.Backbone = backbone[resnet]
        self.instance_fc = FC(backbone_fc_dim, feature_dim)
        self.GDN = GDN(feature_dim, groups)
        self.group_fc = nn.ModuleList([FC(backbone_fc_dim, feature_dim) for i in range(groups)])
        self.feature_dim = feature_dim

    def forward(self, x):
        B = x.shape[0]
        # x = self.Backbone(x)  # (B,4096)
        instacne_representation = self.instance_fc(x)

        # GDN
        group_inter, group_prob = self.GDN(instacne_representation)
        # print(group_prob)
        # group aware repr
        v_G = [Gk(x) for Gk in self.group_fc]  # (B,512)

        # self distributed labeling
        group_label_p = group_prob.data
        group_label_E = group_label_p.mean(dim=0)
        group_label_u = (group_label_p - group_label_E.unsqueeze(dim=-1).expand(self.groups, B).T) / self.groups + (
                1 / self.groups)
        group_label = torch.argmax(group_label_u, dim=1).data

        # group ensemble
        group_mul_p_vk = list()
        if self.mode == 'S':
            for k in range(self.groups):
                Pk = group_prob[:, k].unsqueeze(dim=-1).expand(B, self.feature_dim)
                group_mul_p_vk.append(torch.mul(v_G[k], Pk))
            group_ensembled = torch.stack(group_mul_p_vk).sum(dim=0)
        # instance , group aggregation
        final = instacne_representation + group_ensembled
        return group_inter, final, group_prob, group_label



from collections import OrderedDict
class decoder(nn.Module):
    def __init__(self, in_channels=2048):
        super(decoder, self).__init__()
        layer0 = nn.Sequential(OrderedDict([
            ('conv0', nn.ConvTranspose2d(in_channels, 512, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(512)),
            ('relu0', nn.LeakyReLU(0.1)),
            ('drop0', nn.Dropout(p=0.5)),
        ]))
        layer1 = nn.Sequential(OrderedDict([
            ('conv0', nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(128)),
            ('relu0', nn.LeakyReLU(0.1)),
            ('drop0', nn.Dropout(p=0.5)),
        ]))
        layer2 = nn.Sequential(OrderedDict([
            ('conv0', nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(64)),
            ('relu0', nn.LeakyReLU(0.1)),
            ('drop0', nn.Dropout(p=0.5)),
        ]))
        layer3 = nn.Sequential(OrderedDict([
            ('conv0', nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(32)),
            ('relu0', nn.LeakyReLU(0.1)),
            ('drop0', nn.Dropout(p=0.5)),
        ]))
        layer4 = nn.Sequential(OrderedDict([
            ('conv0', nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(3)),
        ]))
        layer0.apply(weights_init_kaiming)
        layer1.apply(weights_init_kaiming)
        layer2.apply(weights_init_kaiming)
        layer3.apply(weights_init_kaiming)
        layer4.apply(weights_init_kaiming)

        self.layer0 = layer0
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x




import cv2
import random
class ResNet(nn.Module):
    def __init__(self, num_classes, fc_dims=None, loss=None, dropout_p=None,  **kwargs):
        super(ResNet, self).__init__()
        resnet_ = resnet50(pretrained=True)
        self.loss = loss
        self.layer0 = nn.Sequential(
            resnet_.conv1,
            resnet_.bn1,
            resnet_.relu,
            resnet_.maxpool)
        self.layer1 = resnet_.layer1
        self.layer2 = resnet_.layer2
        self.layer3 = resnet_.layer3
        self.att1 = CBAM_Module(1024)
        self.att2 = CBAM_Module(1024)
        self.att_ori = CBAM_Module(1024)

        layer4 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        layer4.load_state_dict(resnet_.layer4.state_dict())

        self.layer41 = nn.Sequential(copy.deepcopy(layer4))
        self.layer42 = nn.Sequential(copy.deepcopy(layer4))
        self.layer43 = nn.Sequential(copy.deepcopy(layer4))
        self.layer_ori = nn.Sequential(copy.deepcopy(layer4))

        


        self.res_part1 = Bottleneck(2048, 512)
        self.res_part2 = Bottleneck(2048, 512)
        self.res_part3 = Bottleneck(2048, 512)

        # self.att_module2 = CBAM_Module(2048)
        # self.att_module3 = CBAM_Module(2048)



        self.batch_erase = RandomErasing_v2()
        self.attent_swap = AttentSwap()

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


        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.global_newpool = nn.AdaptiveMaxPool2d((1, 1))

        self.dropout = nn.Dropout(p=0.75)


        self.harm1=HarmAttn(512)
        self.harm2=HarmAttn(1024)

        self.MultiHeads_1 = MultiHeads(feature_dim=2048, groups=32, mode='S', backbone_fc_dim=2048)
        #self.MultiHeads_2 = MultiHeads(feature_dim=2048, groups=32, mode='S', backbone_fc_dim=2048)


        self.embedding_pre=nn.Linear(4096,2)
        self.classifier1 = nn.Linear(2048, num_classes)
        self.classifier2 = nn.Linear(2048, num_classes)
        self.classifier4 = nn.Linear(4096, num_classes)
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



        
    def featuremaps_my(self, x, x_2, erase_x=None,erase_y=None,erase_x_min=None,erase_y_min=None):
        if self.training:
            b = x.size(0)
            x = torch.cat([x, x_2], 0)
           
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

            x_1 = self.layer41(x_1)
                
            x_2 = self.layer42(x_2)

        else:
            x_1 = self.layer41(x)
                
            x_2 = self.layer42(x)

        return x_1, x_2
    

    def featuremaps_my_1(self, x, x_GAN=None, erase_x=None,erase_y=None,erase_x_min=None,erase_y_min=None):
        if self.training:
            b = x.size(0)
            # x_1 = x.clone()
            # x_1, mask= self.batch_erase(x_1,erase_x,erase_y)
            # x_new=x_1.clone()
            # x_2 = x.clone()
            # x_2=self.attent_swap(x_2,erase_x,erase_y,erase_x_min,erase_y_min)
            x_1, _, _ = self.perturb1(x.detach(), 'train')
            x = torch.cat([x, x_1], 0)
           
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
            # x_3 = x[2*b:, :, :, :]

            x_1 = self.layer41(x_1)
            
            x_2 = self.layer42(x_2)

            # x_3 = self.layer43(x_3)
        else:
            x_1 = self.layer41(x)

            x_2 = self.layer42(x)  

            # x_3 = self.layer43(x)              
        if self.training:
            return x_1, x_2#,newimage

        return x_1, x_2
    
    
    def featuremaps(self, x):
        x = self.layer0(x)   # 64, 96, 32
        x = self.layer1(x)   # 256, 96, 32
        x = self.layer2(x)   # 512, 48, 16
        # x_atten1 = self.harm1(x)
        # x = x*x_atten1
        # x = self.layer3(x)   # 1024, 24, 8
        # x_atten2= self.harm2(x)
        # x = x*x_atten2
        # x_1 = self.layer41(x)
        # x_2 = self.layer42(x)
        return x



    def attent_erase(self,outputs,width=384,height=128):
        erase_x=[]
        erase_y=[]

        erase_x_min=[]
        erase_y_min=[]

        outputs = (outputs**2).sum(1)
        b, h, w = outputs.size()
        outputs = outputs.view(b, h*w)
        outputs = F.normalize(outputs, p=2, dim=1)
        outputs = outputs.view(b, h, w)

        for j in range(outputs.size(0)):
            am = outputs[j, ...].detach().cpu().numpy()
            am = cv2.resize(am, (height, width))
            am = 255 * (am - np.min(am)) / (
                        np.max(am) - np.min(am) + 1e-12
            )
            am = np.uint8(np.floor(am))
            m=np.argmax(am)
            r, c = divmod(m, am.shape[1])

            erase_x.append(r)
            erase_y.append(c)

            # half = int(width / 2)
            # am_c = am[half:,:]
            m_min=np.argmin(am)
            # r_min, c_min = divmod(m_min + half * height, am.shape[1])
            r_min, c_min = divmod(m_min, am.shape[1])
            erase_x_min.append(r_min)
            erase_y_min.append(c_min)

        erase_x=torch.tensor(erase_x).cuda()
        erase_y=torch.tensor(erase_y).cuda()
        erase_x_min=torch.tensor(erase_x_min).cuda()
        erase_y_min=torch.tensor(erase_y_min).cuda()
        return erase_x,erase_y,erase_x_min,erase_y_min#,torch.cat(heatmap,dim=0)



       
    def forward(self, x, x_2=None, return_featuremaps=False, state=True, segmentmask=False,epoch=None,returnflag=False):
        if self.training:
            b = x.size(0)
            if return_featuremaps:
                feature = self.featuremaps(x)
                return feature

            f1, f2 = self.featuremaps_my(x, x_2)

            f1 = self.res_part1(f1)

            f2 = self.res_part2(f2)

            if state == False:
                k = int(b/2)
                v1 = self.global_maxpool(f1[0:k])
                v1 = v1.view(v1.size(0), -1)
                v1_1 = self.reduction1(v1)  # 512

                v1_dis = self.global_maxpool(f1[k:])
                v1_dis = v1_dis.view(v1_dis.size(0), -1)
                v1_1_dis = self.reduction1(v1_dis)

                v2 = self.global_maxpool(f2[0:k])
                v2 = v2.view(v2.size(0), -1)
                v2_1 = self.reduction2(v2)

                v2_dis = self.global_maxpool(f2[k:])
                v2_dis = v2_dis.view(v2_dis.size(0), -1)
                v2_1_dis = self.reduction2(v2_dis)

                y1 = self.classifier1(v1_1)
                y2 = self.classifier2(v2_1)
                y1_dis = self.classifier1(v1_1_dis)
                y2_dis = self.classifier2(v2_1_dis)

                return y1, y2, y1_dis, y2_dis


            v1 = self.global_maxpool(f1)
            v1 = v1.view(v1.size(0), -1)
            v1_1 = self.reduction1(v1)  # 512

            v2 = self.global_maxpool(f2)
            v2 = v2.view(v2.size(0), -1)
            v2_1 = self.reduction2(v2)

            v4_1 = torch.cat([v1_1, v2_1], 1)

            y1 = self.classifier1(v1_1)
            y2 = self.classifier2(v2_1)
            # y3 = self.classifier3(v3_1)
            y4 = self.classifier4(v4_1)

            if self.loss == 'softmax':
                return y1, y2, y4
                # return y1, y2, y3, y4, mask_1
            elif self.loss == 'triplet':
                return y1, y2, y4
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))

        if not self.training:
            if return_featuremaps:
                feature = self.featuremaps(x)
                return feature

            f1, f2= self.featuremaps_my(x, x)



            f1 = self.res_part1(f1)

            f2 = self.res_part2(f2)

            if return_featuremaps:
                return f1

            v1 = self.global_maxpool(f1)
            v1 = v1.view(v1.size(0), -1)
            v1_1 = self.reduction1(v1)  # 512

            v2 = self.global_maxpool(f2)
            v2 = v2.view(v2.size(0), -1)
            v2_1 = self.reduction2(v2)

            v1_1 = F.normalize(v1_1, p=2, dim=1)
            v2_1 = F.normalize(v2_1, p=2, dim=1)

            return torch.cat([v1_1, v2_1], 1)


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

