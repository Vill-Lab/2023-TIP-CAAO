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
import copy
import numpy as np
from torch.nn import functional as F
from torch_multi_head_attention import MultiHeadAttention



__all__ = ['bfe']

class BatchDrop(nn.Module):
    """ref: https://github.com/daizuozhuo/batch-dropblock-network/blob/master/models/networks.py
    batch drop mask
    """

    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h - rh)
            sy = random.randint(0, w - rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx + rh, sy:sy + rw] = 0
            x = x * mask
        return x
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
        return x, spatial_att_map



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
    def forward(self, img,erase_x,erase_y,segmentmask,epoch):
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
       
                        break
        return img



class RandomCropping(nn.Module):
    def __init__(self, sl=0.25, sh=0.5, r1=0.5, mean=[0.4914, 0.4822, 0.4465]):
        super(RandomErasing_v2, self).__init__()
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    # img 32,3,384,128
    def forward(self, img,erase_x,erase_y,segmentmask,epoch):
        for i in range(img.size(0)):
            for attempt in range(10000000000):
                area = img.size()[2] * img.size()[3]
                target_area = random.uniform(self.sl, self.sh) * area
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
       
                        break
        return img





class AdversarialErasing(nn.Module):
    def __init__(self, sl=0.25, sh=0.5, r1=0.5, mean=[0.4914, 0.4822, 0.4465]):
        super(RandomErasing_v3, self).__init__()
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.probability=0.2


    # img 32,3,384,128
    def forward(self, img,erase_x,erase_y,segmentmask,epoch):
        new = img.convert("L")   # Convert from here to the corresponding grayscale image
        print(new.shape)       
        for i in range(img.size(0)):
            for attempt in range(10000000000):
                area = img.size()[2] * img.size()[3]
                target_area = random.uniform(self.sl, self.sh) * area
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
       
                        break
        return img


class RandomErasing_v3(nn.Module):
    def __init__(self, sl=0.25, sh=0.5, r1=0.5, mean=[0.4914, 0.4822, 0.4465]):
        super(RandomErasing_v3, self).__init__()
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.probability=0.2


    # img 32,3,384,128
    def forward(self, img,erase_x,erase_y,segmentmask,epoch):
        new = img.convert("L")   # Convert from here to the corresponding grayscale image
        print(new.shape)       
        for i in range(img.size(0)):
            for attempt in range(10000000000):
                area = img.size()[2] * img.size()[3]
                target_area = random.uniform(self.sl, self.sh) * area
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
       
                        break
        return img

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
        layer4 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        layer4.load_state_dict(resnet_.layer4.state_dict())

        self.layer41 = nn.Sequential(copy.deepcopy(layer4))
        self.layer42 = nn.Sequential(copy.deepcopy(layer4))

        self.res_part1 = Bottleneck(2048, 512)
        self.res_part2 = Bottleneck(2048, 512)

        self.att_module2 = CBAM_Module(2048)

        self.batch_erase = RandomErasing_v2()

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



        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))


        self.embedding_pre=nn.Linear(4096,2)
        self.classifier1 = nn.Linear(2048, num_classes)
        self.classifier2 = nn.Linear(2048, num_classes)
        self.classifier3 = nn.Linear(4096, num_classes)





        self.harm3=HarmAttn(512)
        self.harm4=HarmAttn(1024)



        nn.init.normal_(self.classifier1.weight, 0, 0.01)
        if self.classifier1.bias is not None:
            nn.init.constant_(self.classifier1.bias, 0)
        nn.init.normal_(self.classifier2.weight, 0, 0.01)
        if self.classifier2.bias is not None:
            nn.init.constant_(self.classifier2.bias, 0)
        nn.init.normal_(self.classifier3.weight, 0, 0.01)
        if self.classifier3.bias is not None:
            nn.init.constant_(self.classifier3.bias, 0)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        return mu + epsilon * std
    def patch_shuffle(self,images):
        height=images.shape[2]
        for i in range(int(images.shape[0]/2)):
            image_shuffle=images[0].clone()
            index=random.randint(0,images.shape[0]-1)
            image_shuffle[:,int(height/2):,:]=images[index,:,int(height/2):,:]
            if i==0:
                newimage=image_shuffle.unsqueeze(0)
            else:
                newimage=torch.cat([newimage,image_shuffle.unsqueeze(0)],0)
        return newimage.cuda()

    def featuremaps_my(self, x,erase_x=None,erase_y=None,segmentmask=False,epoch=None):
        if self.training:
            b = x.size(0)
            x_1 = x.clone()
            x_1= self.batch_erase(x_1,erase_x,erase_y,segmentmask,epoch)
            x_new=x_1.clone()
            x = torch.cat([x, x_1], 0)
        x = self.layer0(x)   # 64, 96, 32
        x = self.layer1(x)   # 256, 96, 32
        x = self.layer2(x)   # 512, 48, 16
        x_atten3 = self.harm3(x)
        x = x*x_atten3
        x = self.layer3(x)   # 1024, 24, 8
        x_atten4= self.harm4(x)
        x = x*x_atten4

        if self.training:
            x_1 = x[:b, :, :, :]
            x_2 = x[b:, :, :, :]
            x_1 = self.layer41(x_1)
            x_2,_ = self.att1(x_2)
            x_2 = self.layer42(x_2)
        else:
            x_1 = self.layer41(x)
            x_2,_ = self.att1(x)
            x_2 = self.layer42(x_2)  
        if self.training:
            return x_1, x_2,x_new

        return x_1, x_2
    def featuremaps(self, x):
        x = self.layer0(x)   # 64, 96, 32
        x = self.layer1(x)   # 256, 96, 32
        x = self.layer2(x)   # 512, 48, 16
        x = self.layer3(x)   # 1024, 24, 8
        x_1=x.clone()
        x = self.layer41(x)
        return x,x_1
    def attent_erase(self,outputs,width=384,height=128):
        erase_x=[]
        erase_y=[]
        #heatmap=[]
        for j in range(outputs.size(0)):
            am = outputs[j, ...].detach().cpu().numpy()
            am = cv2.resize(am, (width, height))
            am = 255 * (am - np.min(am)) / (
                        np.max(am) - np.min(am) + 1e-12
            )
            am = np.uint8(np.floor(am))
            #heatmap.append(am)
            m=np.argmax(am)
            r, c = divmod(m, am.shape[1])
            erase_x.append(r)
            erase_y.append(c)

            

        erase_x=torch.tensor(erase_x).cuda()
        erase_y=torch.tensor(erase_y).cuda()
        return erase_x,erase_y#,torch.cat(heatmap,dim=0)
    def mapdrop(self,featmap):
        mask=[]
        featmap_drop=featmap.clone()
        for j in range(featmap.size(0)):
            am = featmap[j, ...].detach()
            am=torch.sum(am,dim=0)
            am=am.cpu().numpy()
            m=np.argmax(am)
            r, c = divmod(m, am.shape[1])
            #print(r)
            #if r-1>=0 and r+1<= featmap.shape[2]:
            featmap_drop[j, : , r, :] = random.uniform(0, 1)
            #if r-1<0 and r+1<= featmap.shape[2]:
            #    featmap_drop[j, : , 0:r+2, :] = 0
            #if r-1>=0 and r+1> featmap.shape[2]:
            #    featmap_drop[j, : , r-2:, :] = 0
        return featmap_drop
    def color_erase(self,images):
        images_new=images
        index=[0,1,2]
        for i in range(images.shape[0]):
            index_shuffle=index
            random.shuffle(index_shuffle)
            images_new[i,0,:,:]=images[i,index_shuffle[0],:,:]
            images_new[i,1,:,:]=images[i,index_shuffle[1],:,:]
            images_new[i,2,:,:]=images[i,index_shuffle[2],:,:]
        return images_new
    def patch_erase(self,images):
        images_new=images
        index=[0,1,2]
        for i in range(images.shape[0]):
            index_shuffle=index
            random.shuffle(index_shuffle)
            images_new[i,0,:,:]=images[i,index_shuffle[0],:,:]
            images_new[i,1,:,:]=images[i,index_shuffle[1],:,:]
            images_new[i,2,:,:]=images[i,index_shuffle[2],:,:]
        return images_new   
    def forward(self, x, return_featuremaps=False,segmentmask=False,epoch=None):
        if self.training:
            fmap,fmap_layer3= self.featuremaps(x)
            erase_x,erase_y=self.attent_erase(fmap)
            ferase_1,ferase_2,x_new=self.featuremaps_my(x,erase_x,erase_y,segmentmask,epoch=epoch)

            fmap_new=ferase_2.clone()
            erase_x_new,erase_y_new=self.attent_erase(fmap_new)
            ferase_1_new,ferase_2_new,x_new_new=self.featuremaps_my(x_new,erase_x_new,erase_y_new,segmentmask,epoch=epoch)
            

            f1=ferase_1
            f2=ferase_2
            f1 = self.res_part1(f1)
            f2 = self.res_part2(f2)
            f2,predmap2 = self.att_module2(f2)
            if return_featuremaps:
                return f2
            v1 = self.global_avgpool(f1)
            v1 = v1.view(v1.size(0), -1)
            v1_1 = self.reduction1(v1)  # 512
         

            v2 = self.global_maxpool(f2)
            v2 = v2.view(v2.size(0), -1)
            v2_1 = self.reduction2(v2)
            v4_1 = torch.cat([v1_1, v2_1], 1)



            y1 = self.classifier1(v1_1)
            y2 = self.classifier2(v2_1)
            y3 = self.classifier3(v4_1)




            f1_new=ferase_1_new
            f2_new=ferase_2_new

            f1_new = self.res_part1(f1_new)
            f2_new = self.res_part2(f2_new)
            f2_new,predmap2_new = self.att_module2(f2_new)

            v1_new = self.global_avgpool(f1_new)
            v1_new = v1_new.view(v1_new.size(0), -1)
            v1_1_new = self.reduction1(v1_new)  # 512

            v2_new = self.global_maxpool(f2_new)
            v2_new = v2_new.view(v2_new.size(0), -1)
            v2_1_new = self.reduction2(v2_new)

            v4_1_new = torch.cat([v1_1_new, v2_1_new], 1)


            y1_new = self.classifier1(v1_1_new)
            y2_new = self.classifier2(v2_1_new)
            y3_new = self.classifier3(v4_1_new)


            if self.loss == 'softmax':
                return y1, y2, y3,y1_new, y2_new, y3_new#,mu,logvar#,mu1,mu2,logvar1,logvar2,mu1_new,mu2_new,logvar1_new,logvar2_new
            elif self.loss == 'triplet':
                return y1, y2, y3
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))

        if not self.training:
            f1, f2= self.featuremaps_my(x)

            f1 = self.res_part1(f1)
            f2 = self.res_part2(f2)
            f2,predmap2= self.att_module2(f2) 

            if return_featuremaps:
                return f2

            v1 = self.global_avgpool(f1)
            v1 = v1.view(v1.size(0), -1)
            v1_1 = self.reduction1(v1)  # 512

            v2 = self.global_maxpool(f2)
            v2 = v2.view(v2.size(0), -1)
            v2_1 = self.reduction2(v2)


            v1_1 = F.normalize(v1_1, p=2, dim=1)
            v2_1 = F.normalize(v2_1, p=2, dim=1)
            v4_1 = torch.cat([v1_1, v2_1], 1)

            return v4_1


# ResNet
def bfe(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        fc_dims=None,
        loss=loss,
        dropout_p=None,
        **kwargs
    )
    return model

