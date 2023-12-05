import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
from .spectral import SpectralNorm
from .opts import get_opts, Imagenet_mean, Imagenet_stddev
from torchvision.models.resnet import resnet50, Bottleneck
import numpy as np
import math
import cv2
import copy
from torch.autograd import Variable

NUM_OPS = 24 #16
NUM_CHANNELS = 3


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


class MLP(nn.Module):
    def __init__(self, n_i, n_h, n_o, n_l):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(n_i, n_h)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(n_h, n_o)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(n_o, n_l)
        self.bn1 = nn.BatchNorm1d(n_h)
        self.bn2 = nn.BatchNorm1d(n_o)
    def forward(self, input):
        return self.linear3(self.relu2(self.bn2(self.linear2(self.relu1(self.bn1(self.linear1(input)))))))

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class Generator(nn.Module):
  def __init__(self, input_nc, output_nc, ngf, norm='bn', n_blocks=4):
    super(Generator, self).__init__()
    # self.maxpool = nn.AdaptiveMaxPool2d((12, 4))
    # self.convnet = nn.Sequential(nn.Conv2d(1024, 128, 3, stride=1, padding=1),
    #                                  nn.ReLU(), nn.Conv2d(128, 64, 3, stride=1, padding=1),
    #                                  nn.ReLU(), nn.Conv2d(64, 32, 3, stride=1, padding=1),
    #                                  nn.ReLU())
    # self.linear_pos = nn.Linear(32 * 12 * 4, NUM_OPS)
    # self.apply(weight_init)
    self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
    self.linear_pos = MLP(24 * 8 + 2048, 512, 256, NUM_OPS)

  def forward(self, inputs, features=None):
    # x = (features.detach() **2).sum(1)
    # b, h, w = x.size()
    # mask = torch.ones(b, 1, 384, 128).cuda()
    # x = x.view(b, h*w)
    # x = F.normalize(x, p=2, dim=1)
    # x = x.view(b, h, w)
    # for j in range(x.size(0)):
    #     actmap = x[j, ...].cpu().numpy()
    #     meank = actmap.mean()
    #     mask_temp = torch.from_numpy(cv2.resize(actmap, (128, 384))).cuda()
    #     mask_temp[mask_temp < meank] = 0
    #     mask_temp[mask_temp >= meank] = 1
    #     mask[j, 0] = mask_temp
    # np.savetxt("/media/tongji/data/qzf/Paritalreid-Check/pos/" + 'pos_复用.txt', mask[0,0].cpu().numpy())
    # x = self.layer0(inputs.detach())
    # x = self.layer1(x)
    x_position = (features.detach() **2).sum(1)
    x_feature = self.maxpool(features.detach())
    # x = self.convnet(x)
    # x_diff = x - x.mean(dim = 0)
    # x = self.layer41(features.detach())
    # x = self.maxpool(features.detach())
    x_position = x_position.view(x_position.size()[0], -1)
    x_feature = x_feature.view(x_feature.size()[0], -1)
    x_position = F.normalize(x_position, p=2, dim=1)
    x_feature = F.normalize(x_feature, p=2, dim=1)
    x = torch.cat([x_position, x_feature], dim=1)

    # x = x.view(b, h*w)
    # x = (x **2).sum(1)
    # b, h, w = x.size()
    # x = x.view(b, h*w)
    # x = F.normalize(x, p=2, dim=1)
    output_pos = self.linear_pos(x)
    
    return output_pos

class Layer_controller(nn.Module):
  def __init__(self, input_nc, output_nc, ngf, norm='bn', n_blocks=4):
    super(Layer_controller, self).__init__()
    resnet_ = resnet50(pretrained=True)
    self.embedding = nn.Embedding(NUM_CHANNELS, 64) # (# of operation) + (# of magnitude) 
    self.linear_channel = nn.Linear(64, NUM_CHANNELS)
    self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

  def forward(self):
    inp = Variable(torch.zeros(1, requires_grad = False).cuda())
    inp = inp.long()
    inp = self.embedding(inp)
    output_channel = self.linear_channel(inp)
    # pos = torch.empty([b, 2])
    # x = self.layer0(inputs)   # 64, 96, 32
    # x = self.layer1(x)   # 256, 96, 32
    # x = self.layer2(x)   # 512, 48, 16
    
    return output_channel.sum(0)




# Define a resnet block
class ResnetBlock(nn.Module):
  def __init__(self, dim, norm_layer, use_bias):
    super(ResnetBlock, self).__init__()
    self.conv_block = self.build_conv_block(dim, norm_layer, use_bias)

  def build_conv_block(self, dim, norm_layer, use_bias):
    conv_block = []
    for i in range(2):
      conv_block += [nn.ReflectionPad2d(1)]
      conv_block += [SpectralNorm(nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias)), norm_layer(dim)]
      if i < 1: 
        conv_block += [nn.ReLU(True)]
    return nn.Sequential(*conv_block)

  def forward(self, x):
    out = x + self.conv_block(x)
    return out

def weights_init(m):
  classname = m.__class__.__name__
  # print(dir(m))
  if classname.find('Conv') != -1:
    if 'weight' in dir(m): 
      m.weight.data.normal_(0.0, 1)
  elif classname.find('BatchNorm2d') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)


def L_norm(delta, mode='train'):
    delta.data += 1 
    delta.data *= 0.5

    delta.data[:,0,:,:] = (delta.data[:,0,:,:] - Imagenet_mean[0]) / Imagenet_stddev[0]

    for i in range(32):  #batchsize = 32
        # do per channel l_inf normalization
        try:
            l_inf_channel = delta[i,0,:,:].data.abs().max()
             # l_inf_channel = torch.norm(delta[i,ci,:,:]).data
            mag_in_scaled_c = 16.0/(255.0*Imagenet_stddev[0])
            delta[i,0,:,:].data *= np.minimum(1.0, mag_in_scaled_c / l_inf_channel.cpu()).float().cuda()
        except IndexError:
            break
    return delta
