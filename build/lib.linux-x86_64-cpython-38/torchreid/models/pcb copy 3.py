from __future__ import absolute_import
from __future__ import division

__all__ = ['pcb_p6', 'pcb_p4']

from torch import nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
import torch
import random
import copy


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DimReduceLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, nonlinear):
        super(DimReduceLayer, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        if nonlinear == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif nonlinear == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


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
        #channel attention
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
        #spatial attention
        module_input = x
        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        x = torch.cat((avg, mx), 1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x
        return x


class PartErasing(nn.Module):
    def __init__(self, h_ratio=0.25, w_ratio=1, Threshold=1, mean=[0.4914, 0.4822, 0.4465]):
        super(PartErasing, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio
        self.it = 0
        self.Threshold = Threshold
        self.sx = None
        self.sy = None
        self.mean = mean

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            if self.it % self.Threshold == 0:
                self.sx = random.randint(0, h - rh)
                self.sy = random.randint(0, w - rw)
            self.it += 1
            x[:, 0, self.sx:self.sx + rh, self.sy:self.sy + rw] = self.mean[0]
            x[:, 1, self.sx:self.sx + rh, self.sy:self.sy + rw] = self.mean[1]
            x[:, 2, self.sx:self.sx + rh, self.sy:self.sy + rw] = self.mean[2]
        return x


class BatchDrop(nn.Module):
    def __init__(self, h_ratio=0.3, w_ratio=1, Threshold=1):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio
        self.it = 0
        self.Threshold = Threshold
        self.sx = None
        self.sy = None

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            if self.it % self.Threshold == 0:
                self.sx = random.randint(0, h - rh)
                self.sy = random.randint(0, w - rw)
            self.it += 1
            mask = x.new_ones(x.size())
            mask[:, :, self.sx:self.sx + rh, self.sy:self.sy + rw] = 0
            x = x * mask
        return x


class PCB(nn.Module):
    """Part-based Convolutional Baseline.

    Reference:
        Sun et al. Beyond Part Models: Person Retrieval with Refined
        Part Pooling (and A Strong Convolutional Baseline). ECCV 2018.

    Public keys:
        - ``pcb_p4``: PCB with 4-part strips.
        - ``pcb_p6``: PCB with 6-part strips.
    """
    
    def __init__(self, num_classes, loss, block, layers,
                 parts=4,
                 reduced_dim=256,
                 nonlinear='relu',
                 **kwargs):
        self.inplanes = 64
        super(PCB, self).__init__()
        self.loss = loss
        self.parts = parts
        self.feature_dim = 512 * block.expansion
        
        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.layer40 = nn.Sequential(copy.deepcopy(self.layer4))
        self.layer41 = nn.Sequential(copy.deepcopy(self.layer4))

        self.res_part0 = Bottleneck(2048, 512)
        self.res_part1 = Bottleneck(2048, 512)
        # self.att_module0 = CBAM_Module(2048)
        self.att_module1 = CBAM_Module(2048)

        self.batch_drop = BatchDrop()

        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        # pcb layers
        self.parts_avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.conv5 = DimReduceLayer(512 * block.expansion, reduced_dim, nonlinear=nonlinear)
        self.feature_dim = reduced_dim
        self.classifier = nn.ModuleList([nn.Linear(self.feature_dim, num_classes) for _ in range(self.parts)])

        # new layers        
        self.conv_pool1 = nn.Sequential(
            nn.Conv2d(2048, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv_pool2 = nn.Sequential(
            nn.Conv2d(2048, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv_pool3 = nn.Sequential(
            nn.Conv2d(2048, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv_pool4 = nn.Sequential(
            nn.Conv2d(2048, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        if self.training:
            b = x.size(0)
            x1 = x.clone() 
            x1 = self.batch_drop(x1)
            x = torch.cat([x, x1], 0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.pc1(x)
        x = self.layer3(x)
        # x = self.att1(x)
        if self.training:
            x_0 = x[:b, :, :, :]
            x_1 = x[b:, :, :, :]
            x_0 = self.layer40(x_0)
            x_1 = self.layer41(x_1)
        else:
            x_0 = self.layer40(x)
            x_1 = self.layer41(x)

        return x_0, x_1

    def forward(self, x):

        N =  x.size(0)
        f0, f1 = self.featuremaps(x)        

        # pcb layers
        # f0 = self.res_part0(f0)
        v_g = self.parts_avgpool(f0)    
        
        v_g = self.dropout(v_g)
        v_h = self.conv5(v_g)
        p1 = v_h[:,:,0,:].view(N, -1)
        p2 = v_h[:,:,1,:].view(N, -1)
        p3 = v_h[:,:,2,:].view(N, -1)
        p4 = v_h[:,:,3,:].view(N, -1)

        y = []
        for i in range(self.parts):
            v_h_i = v_h[:, :, i, :]
            v_h_i = v_h_i.view(v_h_i.size(0), -1)
            y_i = self.classifier[i](v_h_i)
            y.append(y_i)

        # new layers
        f_l = self.res_part1(f1)
        f_l = self.att_module1(f_l)
        t1 = self.global_avgpool(self.conv_pool1(f_l)).squeeze(3).squeeze(2)
        t2 = self.global_avgpool(self.conv_pool2(f_l)).squeeze(3).squeeze(2)
        t3 = self.global_avgpool(self.conv_pool3(f_l)).squeeze(3).squeeze(2)
        t4 = self.global_avgpool(self.conv_pool4(f_l)).squeeze(3).squeeze(2)

        if not self.training:
            t1 = F.normalize(t1, p=2, dim=1) 
            t2 = F.normalize(t2, p=2, dim=1)
            t3 = F.normalize(t3, p=2, dim=1)
            t4 = F.normalize(t4, p=2, dim=1) 
            p1 = F.normalize(p1, p=2, dim=1) 
            p2 = F.normalize(p2, p=2, dim=1)
            p3 = F.normalize(p3, p=2, dim=1)
            p4 = F.normalize(p4, p=2, dim=1)      
            return torch.cat([p1, p2, p3, p4, t1, t2, t3, t4], 1)
        
        if self.loss == 'softmax':
            return y, [p1, p2, p3, p4], [t1, t2, t3, t4]
        elif self.loss == 'triplet':
            return y, [p1, p2, p3, p4], [t1, t2, t3, t4]

        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.
    
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


def pcb_p6(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = PCB(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        parts=6,
        reduced_dim=256,
        nonlinear='relu',
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


def pcb_p4(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = PCB(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        parts=4,
        reduced_dim=256,
        nonlinear='relu',
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model