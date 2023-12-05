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
        self.att2 = CBAM_Module(1024)
        self.half_feature=1024
        self.halfpart=0.4
        layer4 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        layer4.load_state_dict(resnet_.layer4.state_dict())

        self.layer41 = nn.Sequential(copy.deepcopy(layer4))
        self.layer42 = nn.Sequential(copy.deepcopy(layer4))
        self.layer43 = nn.Sequential(copy.deepcopy(layer4))

        self.res_part1 = Bottleneck(2048, 512)
        self.res_part2 = Bottleneck(2048, 512)
        self.res_part3 = Bottleneck(2048, 512)

        self.att_module2 = CBAM_Module(2048)
        self.att_module3 = CBAM_Module(2048)


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


        self.reduction3 = nn.Sequential(
            nn.Linear(2048, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )
        self.reduction3.apply(weights_init_kaiming)



        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.global_maxpool_drop = nn.FractionalMaxPool2d(kernel_size=2,output_size=(1, 1))

        self.dropout = nn.Dropout(p=0.75)


        self.bn_final=nn.BatchNorm1d(4096)
        self.bn_bef1=nn.BatchNorm1d(2048)
        self.bn_bef2=nn.BatchNorm1d(2048)


        self.embedding_pre=nn.Linear(4096,2)
        self.classifier1 = nn.Linear(2048, num_classes)
        self.classifier2 = nn.Linear(2048, num_classes)
        
        self.classifier3 = nn.Linear(4096+2048, num_classes)
        self.classifier4 = nn.Linear(2048, num_classes)



        drop_ratio = 0.4
        self.mu_head = nn.Sequential(
            nn.Dropout(p=drop_ratio),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048, eps=2e-5))

        # use logvar instead of var !!!
        self.logvar_head = nn.Sequential(
            nn.Dropout(p=drop_ratio),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048, eps=2e-5))





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
        for i in range(images.shape[0]):
            image_shuffle=images[0].clone()
            index=random.randint(0,images.shape[0]-1)
            image_shuffle[:,int(height/2):,:]=images[index,:,int(height/2):,:]
            if i==0:
                newimage=image_shuffle.unsqueeze(0)
            else:
                newimage=torch.cat([newimage,image_shuffle.unsqueeze(0)],0)
        return newimage.cuda()

    def featuremaps_my(self, x,erase_x=None,erase_y=None,erase_x_layer3=None,erase_y_layer3=None,segmentmask=False,epoch=None):
        if self.training:
            b = x.size(0)
            x_1 = x.clone()
            x_1= self.batch_erase(x_1,erase_x,erase_y,segmentmask,epoch)
            x_2 = x.clone()
            x_2= self.color_erase(x_2) #self.batch_erase(x_2,erase_x_layer3,erase_y_layer3,segmentmask,epoch)
            x_new=x_1.clone()
            x = torch.cat([x, x_1,x_2], 0)
        x = self.layer0(x)   # 64, 96, 32
        x = self.layer1(x)   # 256, 96, 32
        x = self.layer2(x)   # 512, 48, 16
        x = self.layer3(x)   # 1024, 24, 8
        if self.training:
            
            x_1 = x[:b, :, :, :]
            x_2 = x[b:2*b, :, :, :]
            x_layer3=x_2.clone()
            x_3 = x[2*b:, :, :, :]
            x_1 = self.layer41(x_1)
            x_2,_ = self.att1(x_2)
            x_2 = self.layer42(x_2)

            x_3,_ = self.att2(x_3)              
            x_3 = self.layer43(x_3)
        else:
            x_1 = self.layer41(x)
            x_2,_ = self.att1(x)
            x_2 = self.layer42(x_2)  
            x_3,_ = self.att2(x)              
            x_3 = self.layer43(x_3) 
        if self.training:
            return x_1, x_2, x_3,x_new,x_layer3

        return x_1, x_2,x_3
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
        images_new[:,0,:,:]=images[:,2,:,:]
        images_new[:,1,:,:]=images[:,0,:,:]
        images_new[:,2,:,:]=images[:,1,:,:]
        return images_new

       
    def forward(self, x, return_featuremaps=False,segmentmask=False,epoch=None):
        if self.training:
            fmap,fmap_layer3= self.featuremaps(x)
            erase_x,erase_y=self.attent_erase(fmap)
            erase_x_layer3,erase_y_layer3=self.attent_erase(fmap_layer3)
            ferase_1,ferase_2,ferase_3,x_new,x_new_layer3=self.featuremaps_my(x,erase_x,erase_y,erase_x_layer3,erase_y_layer3,segmentmask,epoch=epoch)

            fmap_new_layer3=x_new_layer3.clone()
            fmap_new=ferase_2.clone()
            erase_x_new,erase_y_new=self.attent_erase(fmap_new)
            erase_x_new_layer3,erase_y_new_layer3=self.attent_erase(fmap_new_layer3)

            ferase_1_new,ferase_2_new,ferase_3_new,x_new_new,_=self.featuremaps_my(x_new,erase_x_new,erase_y_new,erase_x_new_layer3,erase_y_new_layer3,segmentmask,epoch=epoch)
            

            
            
            f1=ferase_1
            f2=ferase_2
            f3=ferase_3


            f1 = self.res_part1(f1)
            f2 = self.res_part2(f2)
            f2,predmap2 = self.att_module2(f2)
            f3 = self.res_part3(f3)
            f3,predmap3 = self.att_module3(f3)
            if return_featuremaps:
                return f2
            v1 = self.global_avgpool(f1)
            v1 = v1.view(v1.size(0), -1)
            v1_1 = self.reduction1(v1)  # 512
         

            v2 = self.global_maxpool(f2)
            v2 = v2.view(v2.size(0), -1)
            v2_1 = self.reduction2(v2)

            v3 = self.global_maxpool_drop(f3)
            v3 = v3.view(v3.size(0), -1)
            v3_1 = self.reduction3(v3)

            v4_1 = torch.cat([v1_1, v2_1,v3_1], 1)
            y1 = self.classifier1(v1_1)
            y2 = self.classifier2(v2_1)
            y4 = self.classifier4(v3_1)
            y3 = self.classifier3(v4_1)




            f1_new=ferase_1_new
            f2_new=ferase_2_new
            f3_new=ferase_3_new

            f1_new = self.res_part1(f1_new)
            f2_new = self.res_part2(f2_new)
            f2_new,predmap2_new = self.att_module2(f2_new)
            f3_new = self.res_part3(f3_new)
            f3_new,predmap3_new = self.att_module3(f3_new)

            v1_new = self.global_avgpool(f1_new)
            v1_new = v1_new.view(v1_new.size(0), -1)
            v1_1_new = self.reduction1(v1_new)  # 512

            v2_new = self.global_maxpool(f2_new)
            v2_new = v2_new.view(v2_new.size(0), -1)
            v2_1_new = self.reduction2(v2_new)

            v3_new = self.global_maxpool_drop(f3_new)
            v3_new = v3_new.view(v3_new.size(0), -1)
            v3_1_new = self.reduction3(v3_new)

            v4_1_new = torch.cat([v1_1_new, v2_1_new,v3_1_new], 1)
            y1_new = self.classifier1(v1_1_new)
            y2_new = self.classifier2(v2_1_new)
            y4_new = self.classifier4(v3_1_new)
            y3_new = self.classifier3(v4_1_new)


            if self.loss == 'softmax':
                return y1, y2, y3,y4,y1_new, y2_new, y3_new,y4_new#,mu1,mu2,logvar1,logvar2,mu1_new,mu2_new,logvar1_new,logvar2_new
            elif self.loss == 'triplet':
                return y1, y2, y3
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))

        if not self.training:
            f1, f2,f3= self.featuremaps_my(x)

            f1 = self.res_part1(f1)

            f2 = self.res_part2(f2)
            f2,predmap2= self.att_module2(f2) 

            f3 = self.res_part3(f3)
            f3,predmap3= self.att_module3(f3) 
            if return_featuremaps:
                return f2

            #f3=f2.clone()
            v1 = self.global_avgpool(f1)
            v1 = v1.view(v1.size(0), -1)
            v1_1 = self.reduction1(v1)  # 512

            v2 = self.global_maxpool(f2)
            v2 = v2.view(v2.size(0), -1)
            v2_1 = self.reduction2(v2)

            v3 = self.global_maxpool_drop(f3)
            v3 = v3.view(v3.size(0), -1)
            v3_1 = self.reduction3(v3)


            v1_1 = F.normalize(v1_1, p=2, dim=1)
            v2_1 = F.normalize(v2_1, p=2, dim=1)
            v3_1 = F.normalize(v3_1, p=2, dim=1)
            v4_1 = torch.cat([v1_1, v2_1,v3_1], 1)
            #v3_1 = F.normalize(v3_1, p=2, dim=1)

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

