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

NUM_OPS = 30 #16
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
    self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
    self.linear_pos = MLP(24 * 8 + 2048, 512, 256, NUM_OPS)

  def forward(self, inputs, features=None):
    x_position = (features.detach() **2).sum(1)
    x_feature = self.maxpool(features.detach())

    x_position = x_position.view(x_position.size()[0], -1)
    x_feature = x_feature.view(x_feature.size()[0], -1)
    x_position = F.normalize(x_position, p=2, dim=1)
    x_feature = F.normalize(x_feature, p=2, dim=1)
    x = torch.cat([x_position, x_feature], dim=1)

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
