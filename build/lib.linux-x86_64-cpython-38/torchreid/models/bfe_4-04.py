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


#可以以最显著区域为中心裁剪出一块区域来做Partialreid任务，然后思路和注意力擦除类似，也做一个样本蒸馏
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
        self.att_module1 = CBAM_Module(2048)



        self.batch_erase = RandomErasing_v2()
        self.patch_erase=BatchErasing()




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

        self.dropout = nn.Dropout(p=0.75)


        self.bn_final=nn.BatchNorm1d(4096*2)
        self.bn_bef1=nn.BatchNorm1d(2048*2)
        self.bn_bef2=nn.BatchNorm1d(2048*2)


        self.embedding_pre=nn.Linear(4096,2)
        self.classifier1 = nn.Linear(2048*2, num_classes)
        self.classifier2 = nn.Linear(2048*2, num_classes)
        self.classifier3 = nn.Linear(4096*2, num_classes)



        self.reduction1_cast = nn.Sequential(
            nn.Linear(2048, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )
        self.reduction1_cast.apply(weights_init_kaiming)

        self.reduction2_cast = nn.Sequential(
            nn.Linear(2048, 2048, 1),

            nn.BatchNorm1d(2048),
            nn.ReLU()
        )
        self.reduction2_cast.apply(weights_init_kaiming)



        self.global_avgpool_cast = nn.AdaptiveAvgPool2d((1, 1))
        self.global_maxpool_cast = nn.AdaptiveMaxPool2d((1, 1))




        nn.init.normal_(self.classifier1.weight, 0, 0.01)
        if self.classifier1.bias is not None:
            nn.init.constant_(self.classifier1.bias, 0)
        nn.init.normal_(self.classifier2.weight, 0, 0.01)
        if self.classifier2.bias is not None:
            nn.init.constant_(self.classifier2.bias, 0)
        nn.init.normal_(self.classifier3.weight, 0, 0.01)
        if self.classifier3.bias is not None:
            nn.init.constant_(self.classifier3.bias, 0)

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
        x = self.layer3(x)   # 1024, 24, 8

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
            return x_1, x_2, x_new

        return x_1, x_2
    def featuremaps(self, x):
        x = self.layer0(x)   # 64, 96, 32
        x = self.layer1(x)   # 256, 96, 32
        x = self.layer2(x)   # 512, 48, 16
        x = self.layer3(x)   # 1024, 24, 8
        x = self.layer41(x)
        return x
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

       
    def forward(self, x, return_featuremaps=False,segmentmask=False,epoch=None):
        if self.training:
            fmap= self.featuremaps(x)
            erase_x,erase_y=self.attent_erase(fmap)
            ferase_1,ferase_2,x_new=self.featuremaps_my(x,erase_x,erase_y,segmentmask,epoch=epoch)

            
            fmap_new=ferase_2.clone()
            erase_x_new,erase_y_new=self.attent_erase(fmap_new)
            ferase_1_new,ferase_2_new,_=self.featuremaps_my(x_new,erase_x_new,erase_y_new,segmentmask,epoch=epoch)
            """
            heatmap=torch.sum(heatmap,dim=2)
            heatmap=transforms.ToPILImage(heatmap)
            heatmap=heatmap.resize((24,8))
            maskmap=nn.Softmax(fmap)
            newfmap=fmap
            """

            f1=ferase_1
            f2=ferase_2
            f1 = self.res_part1(f1)
            f1,predmap1 = self.att_module1(f1)



            f2 = self.res_part2(f2)
            f2,predmap2 = self.att_module2(f2)
            if return_featuremaps:
                return f2


            f1_cast=f1*predmap2
            f2_cast=f2*predmap1

            v1 = self.global_avgpool(f1)
            v1 = v1.view(v1.size(0), -1)
            v1_1 = self.reduction1(v1)  # 512
            v1_cast = self.global_avgpool_cast(f1_cast)
            v1_cast = v1_cast.view(v1_cast.size(0), -1)
            v1_1_cast = self.reduction1_cast(v1_cast)  # 512
            v1_1=torch.cat([v1_1,v1_1_cast],1)






            v2 = self.global_maxpool(f2)
            v2 = v2.view(v2.size(0), -1)
            v2_1 = self.reduction2(v2)
            v2_cast = self.global_maxpool_cast(f2_cast)
            v2_cast = v2_cast.view(v2_cast.size(0), -1)
            v2_1_cast = self.reduction2_cast(v2_cast)
            v2_1=torch.cat([v2_1,v2_1_cast],1)







            v4_1 = torch.cat([v1_1, v2_1], 1)

            y1 = self.classifier1(v1_1)
            y2 = self.classifier2(v2_1)
            y3 = self.classifier3(v4_1)




            f1_new=ferase_1_new
            f2_new=ferase_2_new

            f1_new = self.res_part1(f1_new)
            f1_new,predmap1_new = self.att_module1(f1_new)


            f2_new = self.res_part2(f2_new)
            f2_new,predmap2_new = self.att_module2(f2_new)

            f1_new_cast=ferase_1_new*predmap2_new
            f2_new_cast=ferase_2_new*predmap1_new

            
            v1_new = self.global_avgpool(f1_new)
            v1_new = v1_new.view(v1_new.size(0), -1)
            v1_1_new = self.reduction1(v1_new)  # 512
            v1_new_cast = self.global_avgpool_cast(f1_new_cast)
            v1_new_cast = v1_new_cast.view(v1_new_cast.size(0), -1)
            v1_new_1_cast = self.reduction1_cast(v1_new_cast)  # 512
            v1_1_new=torch.cat([v1_1_new,v1_new_1_cast],1)



            v2_new = self.global_maxpool(f2_new)
            v2_new = v2_new.view(v2_new.size(0), -1)
            v2_1_new = self.reduction2(v2_new)
            v2_new_cast = self.global_maxpool_cast(f2_new_cast)
            v2_new_cast = v2_new_cast.view(v2_new_cast.size(0), -1)
            v2_new_1_cast = self.reduction2_cast(v2_new_cast)  # 512
            v2_1_new=torch.cat([v2_1_new,v2_new_1_cast],1)






            v4_1_new = torch.cat([v1_1_new, v2_1_new], 1)

            y1_new = self.classifier1(v1_1_new)
            y2_new = self.classifier2(v2_1_new)
            y3_new = self.classifier3(v4_1_new)


            if self.loss == 'softmax':
                return y1, y2, y3,y1_new, y2_new, y3_new
            elif self.loss == 'triplet':
                return y1, y2, y3
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))

        if not self.training:
            f1, f2= self.featuremaps_my(x)



            f1 = self.res_part1(f1)
            f1,predmap1= self.att_module1(f1)



            f2 = self.res_part2(f2)
            f2,predmap2= self.att_module2(f2)
            
            if return_featuremaps:
                return f2

            f1_cast=f1*predmap2
            f2_cast=f2*predmap1

            v1 = self.global_avgpool(f1)
            v1 = v1.view(v1.size(0), -1)
            v1_1 = self.reduction1(v1)  # 512
            v1_cast = self.global_avgpool_cast(f1_cast)
            v1_cast = v1_cast.view(v1_cast.size(0), -1)
            v1_1_cast = self.reduction1(v1_cast)  # 512
            v1_1=torch.cat([v1_1,v1_1_cast],1)

            v2 = self.global_maxpool(f2)
            v2 = v2.view(v2.size(0), -1)
            v2_1 = self.reduction2(v2)
            v2_cast = self.global_maxpool_cast(f2_cast)
            v2_cast = v2_cast.view(v2_cast.size(0), -1)
            v2_1_cast = self.reduction2_cast(v2_cast)
            v2_1=torch.cat([v2_1,v2_1_cast],1)




            v1_1 = F.normalize(v1_1, p=2, dim=1)
            v2_1 = F.normalize(v2_1, p=2, dim=1)

            return torch.cat([v1_1, v2_1], 1)


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