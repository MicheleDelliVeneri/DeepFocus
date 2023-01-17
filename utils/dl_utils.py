import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import astropy 
from astropy.io import fits
import os
import sys
import matplotlib.pyplot as plt
from natsort import natsorted
from PIL import Image
import random
import pandas as pd
from kornia.losses import SSIMLoss
import torch.optim as optim
import wandb
from torchvision.utils import make_grid
import torchio as tio
from torch.autograd import Variable
import multiprocessing as mp
from time import time
import torch.distributed as dist
import datetime
import matplotlib
from matplotlib import gridspec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- DEEEP LEARNING FUNCTIONS AND CLASSES ----------------- #
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return 1 - ssim_map.mean()
    else:
        return 1 - ssim_map.mean(1).mean(1).mean(1)
    
def _ssim3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return 1 - ssim_map.mean()
    else:
        return 1 - ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
    
class SSIM3D(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim3D(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def ssim3D(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim3D(img1, img2, window, window_size, channel, size_average)

class batch_act(nn.Module):
    def __init__(self, in_c, act):
        super(batch_act, self).__init__()
        self.bn = nn.BatchNorm2d(in_c)
        self.act = activation_func(act)
    def forward(self, inputs):
        x = self.act(inputs)
        x = self.bn(x)
        return x

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2)  # dynamic add padding based on the kernel_size

class Conv3dAuto(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2, self.kernel_size[2] // 2)  # dynamic add padding based on the kernel_size

class ConvTran2dAuto(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)  # dynamic add padding based on the kernel_size

class ConvTran3dAuto(nn.ConvTranspose3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2, (self.kernel_size[2] - 1) // 2)  # dynamic add padding based on the kernel_size

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()],
        ['sigmoid', nn.Sigmoid()],
        ['tanh', nn.Tanh()],
    ])[activation]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', dim=2):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.dim = dim
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, conv, expansion=1, downsampling=1, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        if self.dim == 2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
                nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None
        elif self.dim == 3:
            self.shortcut = nn.Sequential(
                nn.Conv3d(self.in_channels, self.expanded_channels, kernel_size=1,
                        stride=self.downsampling, bias=False),
                nn.BatchNorm3d(self.expanded_channels)) if self.should_apply_shortcut else None
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

def conv_bn(in_channels, out_channels, conv, dim=2, *args, **kwargs):
    if dim == 2:
        return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))
    elif dim == 3:
        return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm3d(out_channels))

class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, conv, *args, **kwargs):
        super().__init__(in_channels, out_channels, conv,  *args, **kwargs)
        if self.dim == 2:
            self.blocks = nn.Sequential(
                conv_bn(self.in_channels, self.out_channels, self.conv, bias=False, stride=self.downsampling),
                activation_func(self.activation),
                conv_bn(self.out_channels, self.expanded_channels, self.conv, bias=False),
            )
        elif self.dim == 3:
            self.blocks = nn.Sequential(
                conv_bn(self.in_channels, self.out_channels, self.conv, dim=self.dim, bias=False, stride=self.downsampling),
                activation_func(self.activation),
                conv_bn(self.out_channels, self.expanded_channels, self.conv, dim=self.dim, bias=False),)
    
class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, conv, *args, **kwargs):
        super().__init__(in_channels, out_channels, conv, expansion=4, *args, **kwargs)
        if self.dim == 2:
            self.blocks = nn.Sequential(
                conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
                activation_func(self.activation),
                conv_bn(self.out_channels, self.out_channels, self.conv, stride=self.downsampling),
                activation_func(self.activation),
                conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
            )
        elif self.dim == 3:
            self.blocks = nn.Sequential(
                conv_bn(self.in_channels, self.out_channels, self.conv,  dim=self.dim, kernel_size=1),
                activation_func(self.activation),
                conv_bn(self.out_channels, self.out_channels, self.conv,  dim=self.dim, stride=self.downsampling),
                activation_func(self.activation),
                conv_bn(self.out_channels, self.expanded_channels, self.conv, dim=self.dim,  kernel_size=1),
            )

class ResNetLayer(nn.Module):
    """
    A ResNet layer composed by `n` blocks stacked one after the other
    """
    def __init__(self, in_channels, out_channels, kernel_size, block=ResNetBasicBlock, n=1, dim=2, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        if dim == 2:
            conv = partial(Conv2dAuto, kernel_size=kernel_size)
        elif dim == 3:
            conv = partial(Conv3dAuto, kernel_size=kernel_size)
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, conv, *args, **kwargs, downsampling=downsampling, dim=dim),
            *[block(out_channels * block.expansion, 
                    out_channels, conv, downsampling=1, *args, **kwargs, dim=dim) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], kernel_sizes=[3, 3, 3, 3], 
                 depths=[2,2,2,2], hidden_size=1024,
                 activation='relu', block=ResNetBasicBlock, input_shape=(256, 256),
                 skip_connections=False, 
                 debug=False,
                 dmode='deconvolver',
                 dropout_rate=0.0,
                 *args,**kwargs):
        super().__init__()
        
        self.blocks_sizes = blocks_sizes
        self.depths = depths
        self.kernel_sizes = kernel_sizes
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.skip_connections = skip_connections
        self.debug = debug
        self.dmode = dmode
        self.dropout_rate = dropout_rate
        if block == ResNetBasicBlock:
            self.expansion = 1
        elif block == ResNetBottleNeckBlock:
            self.expansion = 4
        if len(self.input_shape) == 2:
            self.dim = 2
            self.gate = nn.Sequential(
                nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(self.blocks_sizes[0]),
                activation_func(activation),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            self.dense = nn.Sequential(
                    nn.Linear(blocks_sizes[-1] * input_shape[0] // 2**(len(depths) + 1) * input_shape[1] // 2**(len(depths) + 1), self.hidden_size),
                    activation_func(activation))
        elif len(self.input_shape) == 3:
            self.dim = 3
            self.gate = nn.Sequential(
                nn.Conv3d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm3d(self.blocks_sizes[0]),
                activation_func(activation),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
            )
            if self.dropout_rate > 0.0:
                self.dense = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(blocks_sizes[-1] * self.expansion * input_shape[0] // 2**(len(depths) + 1) * input_shape[1] // 2**(len(depths) + 1)* input_shape[2] // 2**(len(depths) + 1), self.hidden_size),
                    activation_func(activation))
            else:
                self.dense = nn.Sequential(
                    nn.Linear(blocks_sizes[-1] * self.expansion * input_shape[0] // 2**(len(depths) + 1) * input_shape[1] // 2**(len(depths) + 1)* input_shape[2] // 2**(len(depths) + 1), self.hidden_size),
                    activation_func(activation))
        self.in_out_block_sizes_kernels = list(zip(blocks_sizes, blocks_sizes[1:], kernel_sizes))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], kernel_sizes[0], n=depths[0], activation=activation, 
                        block=block,  *args, **kwargs, dim=self.dim),
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, kernel_size, n=n, activation=activation, 
                          block=block, *args, **kwargs, dim=self.dim) 
              for (in_channels, out_channels, kernel_size), n in zip(self.in_out_block_sizes_kernels, depths[1:])]       
        ])
            
    def forward(self, x):
        x = self.gate(x)
        if self.debug:
            print('Encoder Gate: ', x.shape)
        if self.skip_connections:
            skip = []
        for i, block in zip(np.arange(len(self.blocks)), self.blocks):
            x = block(x)
            if self.debug:
                print('After Block {}: '.format(i), x.shape)
            if self.skip_connections:
                skip.append(x)
        if self.dmode == 'deconvolver':
            x = torch.flatten(x, start_dim=1)
            x = self.dense(x)
            if self.debug:
                print('Encoder Dense: ', x.shape)
        if self.skip_connections:
            return x, skip[:-1]
        else:
            return x

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x

class ResNetDecoderLayer(nn.Module):
    """
    A ResNet decoder layer composed by upsampling and concatenating with a skip connection
    """
    def __init__(self, in_channels, out_channels, kernel_size, block=ResNetBasicBlock, dim=2, activation='relu', *args, **kwargs):
        super().__init__()
        self.dim = dim
        self.activation = activation
        if self.dim == 2:
            self.interpolate = Interpolate(scale_factor=2, mode='bilinear', align_corners=True)
            tconv = partial(ConvTran2dAuto, kernel_size=kernel_size)
            self.upsample = nn.Sequential(
                tconv(in_channels, out_channels, stride=2, padding=1, output_padding=1),
                #nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_channels),
                activation_func(activation)
            )
            conv = partial(Conv2dAuto, kernel_size=kernel_size)
        elif self.dim == 3:
            self.interpolate = Interpolate(scale_factor=2, mode='trilinear', align_corners=True)
            tconv = partial(ConvTran3dAuto, kernel_size=kernel_size)
            self.upsample = nn.Sequential(
                #nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1, output_padding=1),
                tconv(in_channels, out_channels, stride=2, padding=1, output_padding=1),
                nn.BatchNorm3d(out_channels),
                activation_func(activation)
            )
            conv = partial(Conv3dAuto, kernel_size=kernel_size)
        self.block = block(in_channels + out_channels, out_channels, conv, *args, **kwargs, 
                      dim=self.dim, activation=self.activation)
        self.oblock = block(in_channels, out_channels, conv, *args, **kwargs, 
                      dim=self.dim, activation=self.activation)
    def forward(self, x, skip=None):
        #print('Deconder Input ',  x.shape)
        i = self.interpolate(x)
        x = self.upsample(x)
        x = torch.cat([x, i], dim=1)
        #print('Deconder After Upsample ',  x.shape)
        x = self.block(x)
        #print('Print After Double Channel ', x.shape)
        if skip != None:
            #print('Before concatenate: ',x.shape, skip.shape)
            x = torch.cat([x, skip], dim=1)
            #print('After concatenate: ',x.shape)
            #print(self.block.in_channels, x.shape[1])
            x = self.oblock(x)
        return x

class View(nn.Module):
    def __init__(self, dim,  shape):
        super(View, self).__init__()
        self.dim = dim
        self.shape = shape

    def forward(self, input):
        new_shape = list(input.shape)[:self.dim] + list(self.shape) + list(input.shape)[self.dim+1:]
        return input.view(*new_shape)

nn.Unflatten = View

class ResNetDecoder(nn.Module):
    def __init__(self, output_channels=1, blocks_sizes=[512, 256, 128, 64], 
                 oblocks_sizes=[64, 32, 1],
                 kernel_sizes=[3, 3, 3, 3],
                 okernel_sizes=[3, 3, 3],
                 hidden_size=1024,
                 activation='relu',
                 final_activation='sigmoid',
                 block=ResNetBasicBlock, input_shape=(256, 256),
                 skip_connections=False,
                 debug=False,
                 *args,**kwargs):
        super().__init__()
        
        self.blocks_sizes = blocks_sizes
        self.oblocks_sizes = oblocks_sizes
        self.kernel_sizes = kernel_sizes
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.output_channels = output_channels
        self.skip_connections = skip_connections
        self.activation = activation
        
        self.debug = debug
        if block == ResNetBasicBlock:
            self.expansion = 1
        elif block == ResNetBottleNeckBlock:
            self.expansion = 4
        self.in_out_block_sizes_kernels = list(zip(blocks_sizes[1:], blocks_sizes[2:], kernel_sizes[1:]))
        self.out_block_sizes_kernels = list(zip(oblocks_sizes, oblocks_sizes[1:], okernel_sizes))
        print('Expansion factor: ', self.expansion)
        if len(self.input_shape) == 2:
            self.dim = 2
            self.dense = nn.Sequential(
                    nn.Linear(self.hidden_size, blocks_sizes[0] * self.expansion * input_shape[0] // 2**(len(self.blocks_sizes) + 1) * input_shape[1] // 2**(len(self.blocks_sizes) + 1)),
                    activation_func(activation))

            self.unflatten = nn.Unflatten(dim=1, shape=(blocks_sizes[0] * self.expansion, 
                                input_shape[0] // 2**(len(self.blocks_sizes) + 1), 
                                input_shape[1] // 2**(len(self.blocks_sizes) + 1)))
            self.blocks = nn.ModuleList([ 
                ResNetDecoderLayer(blocks_sizes[0] * self.expansion, blocks_sizes[1], kernel_sizes[0], block=block, *args, **kwargs, 
                                    dim=self.dim, activation=self.activation),
                *[ResNetDecoderLayer(in_channels * self.expansion, 
                                 out_channels * self.expansion, kernel_size, block=block, *args, **kwargs, 
                                 dim=self.dim, activation=self.activation) 
                 for (in_channels, out_channels, kernel_size) in self.in_out_block_sizes_kernels]       
            ])
            self.oblocks = nn.ModuleList([ResNetDecoderLayer(in_channels * self.expansion, 
                                 out_channels, kernel_size, block=block, *args, **kwargs, 
                                 dim=self.dim, activation=self.activation) 
              for (in_channels, out_channels, kernel_size) in self.out_block_sizes_kernels])
            self.out = nn.Sequential(
                nn.Conv2d(self.oblocks_sizes[-1] * self.expansions, self.output_channels, kernel_size=1, padding=0),
                activation_func(final_activation))
        elif len(self.input_shape) == 3:
            self.dim = 3
            self.dense = nn.Sequential(
                nn.Linear(self.hidden_size, blocks_sizes[0] * self.expansion * input_shape[0] // 2**(len(self.blocks_sizes) + 1) * input_shape[1] // 2**(len(self.blocks_sizes) + 1) * input_shape[2] // 2**(len(self.blocks_sizes) + 1)),
                activation_func(activation))
            self.unflatten = nn.Unflatten(dim=1, shape=(blocks_sizes[0] * self.expansion,
                                            input_shape[0] // 2**(len(self.blocks_sizes) + 1),
                                            input_shape[1] // 2**(len(self.blocks_sizes) + 1),
                                            input_shape[2] // 2**(len(self.blocks_sizes) + 1)))
            self.blocks = nn.ModuleList([ 
                ResNetDecoderLayer(blocks_sizes[0] * self.expansion, blocks_sizes[1], kernel_sizes[0], block=block, *args, **kwargs, dim=self.dim),
                *[ResNetDecoderLayer(in_channels * self.expansion, 
                                 out_channels , kernel_size, block=block, *args, **kwargs, dim=self.dim) 
                 for (in_channels, out_channels, kernel_size) in self.in_out_block_sizes_kernels]       
            ])
            self.oblocks = nn.ModuleList([ResNetDecoderLayer(in_channels * self.expansion, 
                                 out_channels, kernel_size, block=block, *args, **kwargs, dim=self.dim) 
              for (in_channels, out_channels, kernel_size) in self.out_block_sizes_kernels])
            self.out = nn.Sequential(
                    nn.Conv3d(self.oblocks_sizes[-1] * self.expansion, self.output_channels, kernel_size=1, padding=0),
                    activation_func(final_activation))

    
    def forward(self, x, skips=[None]):
        x = self.dense(x)
        if self.debug:
            print('After Decoder Dense: ', x.shape)
        x = self.unflatten(x)
        if self.debug:
            print('After Unflattering: ', x.shape)
        if self.skip_connections is True:
            skips = skips[::-1]
        else:
            skip = [None] * len(self.blocks)
        for i,  block, skip in zip(np.arange(len(self.blocks)), self.blocks, skips):
            if self.debug and skip != None:
                print('Before Decoder Block: ', x.shape, skip.shape)
            elif self.debug:
                print('Before Decoder Block: ', x.shape)
            x = block(x, skip)
            if self.debug:
                print('After Decoder Block {}: '.format(i), x.shape)
        for i,  block in zip(np.arange(len(self.oblocks)), self.oblocks):
            x = block(x, None)
            if self.debug:
                print('After Decoder Out Block {}: '.format(i), x.shape)
        x = self.out(x)
        if self.debug:
            print('After Out: ', x.shape)
        return x

class ResNetRegressor(nn.Module):
    def __init__(self, in_features, n_hidden, n_classes, activation='relu', classification=False, debug=False, dropout_rate=0.0, *args, **kwargs):
        super().__init__()
        self.debug = debug
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        if classification is True:
            self.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, n_hidden),
                activation_func(activation),
                nn.Linear(n_hidden, n_classes),
                nn.Softmax(dim=1))
        else:
            self.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, n_hidden),
                activation_func(activation),
                nn.Linear(n_hidden, n_classes))
    def forward(self, x):
        x = self.avg(x)
        if self.debug:
            print('After Avg Pooling: ', x.shape)
        x = x.view(x.size(0), -1)
        if self.debug:
            print('After Flattening: ', x.shape)
        x = self.fc(x)
        if self.debug:
            print('After FC: ', x.shape)
        return x
    
class DeepFocus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, blocks_sizes=[64, 128, 256, 512],
                 oblocks_sizes=[64, 32, 1],
                 encoder_kernel_sizes=[3, 3, 3, 3],
                 depths = [2, 2, 2, 2],
                 decoder_kernel_sizes=[3, 3, 3, 3],
                 output_kernel_sizes=[3, 3, 3],
                 hidden_size=1024,
                 encoder_activation='relu',
                 decoder_activation='relu',
                 encoder_block=ResNetBasicBlock,
                 decoder_block=ResNetBasicBlock,
                 final_activation='sigmoid',
                 input_shape=(256, 256),
                 skip_connections=False,
                 dmode='deconvolver',
                 debug=False,
                 dropout_rate=0.0, 
                 *args,**kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depths = depths
        self.encoder_blocks_sizes = blocks_sizes
        self.decoder_blocks_sizes = blocks_sizes[::-1]
        self.oblocks_sizes = oblocks_sizes
        self.encoder_kernel_sizes = encoder_kernel_sizes
        self.decoder_kernel_sizes = decoder_kernel_sizes
        self.output_kernel_sizes = output_kernel_sizes
        self.hidden_size = hidden_size
        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation
        self.encoder_block = encoder_block
        self.decoder_block = decoder_block
        self.final_activation = final_activation
        self.input_shape = input_shape
        self.skip_connections = skip_connections
        self.debug = debug
        self.dmode = dmode
        self.dropout_rate = dropout_rate
        self.encoder = ResNetEncoder(in_channels=self.in_channels, 
                                     blocks_sizes=self.encoder_blocks_sizes,
                                     kernel_sizes=self.encoder_kernel_sizes,
                                     depths=self.depths,
                                     hidden_size=self.hidden_size,
                                     activation=self.encoder_activation,
                                     block=self.encoder_block,
                                     input_shape=self.input_shape,
                                     skip_connections=self.skip_connections,
                                     debug=self.debug,
                                     dmode=self.dmode,
                                     dropout_rate = self.dropout_rate
                                     )
        if self.dmode == 'deconvolver':
            print('Building Deconvolver')
            self.decoder = ResNetDecoder(output_channels=self.out_channels,
                                     blocks_sizes=self.decoder_blocks_sizes,
                                     oblocks_sizes=self.oblocks_sizes,
                                        kernel_sizes=self.decoder_kernel_sizes,
                                        okernel_sizes=self.output_kernel_sizes,
                                        hidden_size=self.hidden_size,
                                        activation=self.decoder_activation,
                                        final_activation=self.final_activation,
                                        block=self.decoder_block,
                                        input_shape=self.input_shape,
                                        skip_connections=self.skip_connections,
                                        debug=self.debug)
        elif self.dmode == 'regressor':
            print('Building Regressor')
            self.decoder = ResNetRegressor(in_features=self.encoder_blocks_sizes[-1]  * self.encoder_block.expansion,
                                           n_hidden=self.hidden_size,
                                           n_classes=self.out_channels,
                                           activation=self.decoder_activation,
                                           classification=False,
                                           debug=self.debug, 
                                           dropout_rate=self.dropout_rate)
        elif self.dmode == 'classifier':
            print('Building Classifier')
            self.decoder = ResNetRegressor(in_features=self.encoder_blocks_sizes[-1]  * self.encoder_block.expansion,
                                           n_hidden=self.hidden_size,
                                           n_classes=self.out_channels,
                                           activation=self.decoder_activation,
                                           classification=True,
                                           debug=self.debug,
                                           dropout_rate=self.dropout_rate)
    def forward(self, x):
        if self.skip_connections:
            lat, skips = self.encoder(x)
            x = self.decoder(lat, skips)
            return x
        else:
            lat = self.encoder(x)
            x = self.decoder(lat)
            return x

# ----------------------- BUILDING FUNCTIONS ----------------------- #  

def build_loss(criterion_name, input_shape):
        if len(criterion_name) > 1:
            criterion = []
            for crit in criterion_name:
                if crit == 'BCE':
                    criterion.append(nn.BCELoss())
                elif crit == 'MSE':
                    criterion.append(nn.MSELoss())
                elif crit == 'L1':
                    criterion.append(nn.L1Loss())
                elif crit == 'SSIM':
                    if len(input_shape) == 2:
                        criterion.append(SSIMLoss(window_size=3))
                    elif len(input_shape) == 3:
                        criterion.append(SSIM3D(window_size = 3))
        else:
            criterion = None
            if criterion_name[0] == 'BCE':
                criterion = nn.BCELoss()
            elif criterion_name[0] == 'MSE':
                criterion = nn.MSELoss()
            elif criterion_name[0] == 'L1':
                criterion = nn.L1Loss()
            elif criterion_name[0] == 'SSIM':
                if len(input_shape) == 2:
                    criterion = SSIMLoss(window_size=3)
                elif len(input_shape) == 3:
                    criterion = SSIM3D(window_size = 3)
        return criterion  

def build_optimizer(optimizer_name, model, lr, weight_decay):
    optimizer = None
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer

def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        },
        save_path,
    )

def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, optimizer, epoch

# -----------------------  TNG DATASET DATALOADING UTILS ----------------------- #

def load_fits(inFile):
    hdu_list = fits.open(inFile)
    data = hdu_list[0].data
    hdu_list.close()
    return data

def load_catalogue(path):
    df = pd.read_csv(path, sep='\t')
    return df

def plot_image(x, panel_names, savename=None):
    fig, ax = plt.subplots(nrows=2, ncols=x.shape[0] // 2, figsize=(8 * (x.shape[0] // 2), 8 * (2)))
    n = 0
    for i in range(2):
        for k in range(x.shape[0] // 2):
            img = x[n]
            im = ax[i, k].imshow(img, cmap='magma', interpolation='nearest')
            ax[i, k].set_title(panel_names[n])
            fig.colorbar(im, ax=ax[i, k])
            n += 1
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    else:
        plt.show()
    plt.close()
    n += 1

def TNG_prepare_data(data_path, catalogue_path, val_size=0.2, test_size=0.2, random_state=42):
    filelist = np.array(natsorted(os.listdir(data_path)))
    catalogue = load_catalogue(catalogue_path)
    idlist = catalogue['ID'].values
    image_ids = np.array([int("".join([t for t in tid.split('_')[0] if (t.isdigit())])) for tid in filelist])
    complete_ids = []
    for id in idlist:
        idxs = np.where(image_ids == id)[0].astype(int)
        if len(idxs) // 5 == 26:
            for i in idxs:
                complete_ids.append(i)
    complete_ids = np.array(complete_ids)
    unique_ids = np.unique(image_ids[complete_ids])
    catalogue = catalogue[catalogue['ID'].isin(unique_ids)]
    idlist = catalogue['ID'].values
    train_ids, test_ids = train_test_split(idlist, test_size=test_size, random_state=random_state)
    train_ids, val_ids = train_test_split(train_ids, 
        test_size=(len(idlist) * val_size) / (len(train_ids)), random_state=random_state)
    print('Train/Val/Test Sizes: ', len(train_ids) / len(idlist), len(val_ids) / len(idlist), len(test_ids) / len(idlist))
    return catalogue, filelist, image_ids, complete_ids,  train_ids, val_ids, test_ids

class TNGDataset(Dataset):
    def __init__(self, data_dir, catalogue, filelist, image_ids, file_ids, idxs, 
                 transforms, normalize_output, preprocess,
                 channels, map, show_plots=False):
        self.data_dir = data_dir
        self.filelist = filelist
        self.idxs = idxs
        self.transforms = transforms
        self.epsilon = 0.0001
        self.catalogue = catalogue
        self.image_ids = image_ids
        self.file_ids = file_ids
        self.normalize_output = normalize_output
        self.channels = channels
        self.preprocess = preprocess
        self.map = map
        self.show_plots = show_plots
        self.totensor = A.Compose([
            A.Resize(256, 256),
            ToTensorV2()])
    
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        idxs = np.where(self.image_ids == [self.idxs[idx]])[0].astype(int)
        filenames = self.filelist[idxs]
        properties = self.catalogue[self.catalogue['ID'] == self.idxs[idx]]
        rot = "O" + str(random.randint(1, 5))
        filenames = [filename for filename in filenames if filename.split('_')[1] == rot]
        bands_names = [filename for filename in filenames if "_".join(filename.split('.')[0].split('_')[2:]) in self.channels]
        bands = np.array([load_fits(os.path.join(self.data_dir, filename)) for filename in bands_names])
        target_names = [filename for filename in filenames if "_".join(filename.split('.')[0].split('_')[2:]) in self.map]
        otarget = np.array([load_fits(os.path.join(self.data_dir, filename)) for filename in target_names])
        target = otarget.copy()
        if self.preprocess is not None:
            if self.preprocess == 'log':
                bands = np.log10(bands + self.epsilon)
                target = np.log10(otarget + self.epsilon)
            elif self.preprocess == 'sqrt':
                bands = np.sqrt(bands)
                target = np.sqrt(otarget)
            elif self.preprocess == 'log_sqrt':
                bands = np.sqrt(np.log10(bands + self.epsilon))
                target = np.sqrt(np.log10(otarget + self.epsilon))
            elif self.preprocess == 'sinh':
                target = np.sinh(otarget)
                bands = np.sinh(bands)
        if self.normalize_output is True:
            target = (target - target.min()) / (target.max() - target.min())
        

        if self.transforms is not None:
            transformed = self.transforms(image=np.transpose(bands, axes=(1, 2, 0)), mask=np.transpose(target, axes=(1, 2, 0)))
        else:
            transformed = self.totensor(image=np.transpose(bands, axes=(1, 2, 0)), mask=np.transpose(target, axes=(1, 2, 0)))
        input_ = transformed['image']
        target_ = np.transpose(transformed['mask'], axes=(2, 0, 1))
        if self.show_plots:
            print('Simulation {} rotation {}'.format(self.idxs[idx], rot))       
            fig, axs = plt.subplots(nrows=len(self.channels) // 3, ncols=3, figsize=(20, 20))
            for i, ax in enumerate(axs.flatten()):
                imi = ax.imshow(input_[i], cmap='magma')
                plt.colorbar(imi, ax=ax)
                ax.set_title(self.channels[i])
            plt.show()
            plt.figure(figsize=(4, 4))
            plt.imshow(target_[0], cmap='magma')
            plt.colorbar()
            plt.title(self.map)
            plt.show()

            print('Stellar Mass: ', properties['stellar_mass'].values[0])
            print('Original Fits Target image properties: ', '\nMin: ', otarget.min(),
                '\nMax: ',  otarget.max(), '\nMean: ', otarget.mean(), '\nStd: ', otarget.std(),
                '\nIntegral: ', np.sum(otarget))
        print(input_.shape, target_.shape)
        return input_, target_

# --------------------- ALMA DATASET DATALOADING UTILS --------------------- #

class ALMADataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.root_dir = root_dir
        self.parameters = pd.read_csv(os.path.join(self.root_dir, csv_file))
        self.dirty_list = np.array(natsorted([
            os.path.join(self.root_dir, file) for file in os.listdir(self.root_dir) if 'dirty' in file]))
        self.clean_list = np.array(natsorted([
            os.path.join(self.root_dir, file) for file in os.listdir(self.root_dir) if 'clean' in file]))

    def __len__(self):
        return len(self.dirty_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()      
        dirty_name = self.dirty_list[idx]
        clean_name = self.clean_list[idx]
        clean_cube = fits.getdata(clean_name)
        dirty_cube = fits.getdata(dirty_name)
        db_idx = float(clean_name.split('_')[-1].split('.')[0])
        params = self.parameters.loc[self.parameters.ID == db_idx]
        boxes = np.array(params[["y0", "x0", "y1", "x1"]].values)
        z_ = np.array(params["z"].values)
        fwhm_z = np.array(params["fwhm_z"].values)
        zboxes = np.array([z_ - fwhm_z, z_ + fwhm_z]).astype(int).T
        targets = np.array(params[["x", "y", "fwhm_x", "fwhm_y", "pa", "flux", 'continuum']].values)
        focused = []
        xs = []
        ys = []
        for i, box in enumerate(boxes):
            y_0, x_0, y_1, x_1 = box
            z_0, z_1 = zboxes[i]
            width_x, width_y = x_1 - x_0, y_1 - y_0
            x, y = x_0 + 0.5 * width_x, y_0 + 0.5 * width_y
            source = np.sum(dirty_cube[0][z_0:z_1, int(y) - 32: int(y) + 32, int(x) - 32: int(x) + 32], axis=0)
            xsize, ysize = int(source.shape[0]), int(source.shape[1])
            dx, dy = xsize - 64, ysize - 64
            if dx % 2 == 0:
                left, right = dx // 2, xsize - dx // 2
            else:
                left, right = dx // 2, xsize - dx // 2 - 1
            if dy % 2 == 0:
                bottom, top = dy // 2, ysize - dy // 2
            else:
                bottom, top = dy // 2, ysize - dy // 2 - 1
            source = source[left:right, bottom:top]
            min_, max_ = np.min(source), np.max(source)
            source = (source - min_) / (max_ - min_)
            focused.append(source)
            xs.append(32 - left)
            ys.append(32 - bottom)
        targets[:, 0] = xs
        targets[:, 1] = ys
        focused = torch.from_numpy(np.array(focused)[np.newaxis, :, :])
        sample = {'inputs': focused, 'targets': targets}
        return sample
    
    def collate_fn(self, batch):
        focused, targets  = list(), list()
        for b in batch:
            for b_n in range(len(b['inputs'])):
                focused.append(b['inputs'][:, b_n])
                targets.append(b['targets'][b_n])
        focused = torch.stack(focused, dim=0)
        targets = np.array(targets)
        return focused, targets

def ALMAreader(path):
    data = np.nan_to_num(fits.getdata(path)).transpose(0, 2, 3, 1).astype(np.float32)
    affine = np.eye(4)
    return data, affine

def get_ALMA_dataloaders(root_dir, training_transforms=None, validation_transforms=None, 
                        batch_size=32,  num_workers=4):
    train_dir = root_dir + 'Train'
    valid_dir = root_dir + 'Validation'
    train_dirty_list = np.array(natsorted([
            os.path.join(train_dir, file) for file in os.listdir(train_dir) if 'dirty' in file]))
    train_clean_list = np.array(natsorted([
            os.path.join(train_dir, file) for file in os.listdir(train_dir) if 'clean' in file]))
    valid_dirty_list = np.array(natsorted([
            os.path.join(valid_dir, file) for file in os.listdir(valid_dir) if 'dirty' in file]))
    valid_clean_list = np.array(natsorted([
            os.path.join(valid_dir, file) for file in os.listdir(valid_dir) if 'clean' in file]))
    
    assert len(train_dirty_list) == len(train_clean_list), 'Number of dirty and clean cubes in Training set do not match'
    assert len(valid_dirty_list) == len(valid_clean_list), 'Number of dirty and clean cubes in Validation set do not match'
    
    train_samples = []
    for (dirty_path, clean_path) in zip(train_dirty_list, train_clean_list):
        sample = tio.Subject(
            dirty=tio.ScalarImage(dirty_path, reader=ALMAreader),
            clean=tio.ScalarImage(clean_path, reader=ALMAreader),
        )
        train_samples.append(sample)
    
    valid_samples = []
    for (dirty_path, clean_path) in zip(valid_dirty_list, valid_clean_list):
        
        sample = tio.Subject(
            dirty=tio.ScalarImage(dirty_path, reader=ALMAreader),
            clean=tio.ScalarImage(clean_path, reader=ALMAreader),
        )
        valid_samples.append(sample)
    
    training_dataset = tio.SubjectsDataset(train_samples, transform=training_transforms)
    validation_dataset = tio.SubjectsDataset(valid_samples, transform=validation_transforms)
    if num_workers == 'auto':
        times = []
        nws = []
        print('Automatically determining the best number of workers...')
        print('Found {} CPU cores'.format(mp.cpu_count()))
        for nw in tqdm(range(8, mp.cpu_count(), 2), total=(mp.cpu_count() - 8) //2):
            train_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=nw, pin_memory=True)
            start = time()
            for epoch in range(1, 3):
                for i, data in enumerate(train_dataloader, 0):
                    pass
            end = time()
            print("Finish with:{} second, num_workers={}".format(end - start, nw))
            times.append(end - start)
            nws.append(nw)
        num_workers = nws[np.argmin(times)]
        print("Best num_workers is {}".format(num_workers))

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers, pin_memory=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    return training_dataloader, validation_dataloader

def get_ALMA_test_dataloaders(root_dir, tclean_dir, test_transforms=None,
                        batch_size=32,  num_workers=4, get_tclean=False):
    if get_tclean is True:
        test_dir = root_dir + 'Test'
        ids = [int(x.split('_')[-1].split('.')[0]) for x in os.listdir(test_dir) if 'dirty' in x]
        dirty_list = np.array(natsorted([
            os.path.join(test_dir, file) for file in os.listdir(test_dir) if 'dirty' in file]))
        clean_list = np.array(natsorted([
            os.path.join(test_dir, file) for file in os.listdir(test_dir) if 'clean' in file]))
        tclean_list = np.array(natsorted([os.path.join(tclean_dir, file) for file in os.listdir(tclean_dir) if (
                'tcleaned' in file and int(file.split('_')[-1].split('.')[0]) in ids)]))
        ids = [int(x.split('_')[-1].split('.')[0]) for x in tclean_list]
        test_samples = []
        for (dirty_path, clean_path, tclean_path) in zip(dirty_list, clean_list, tclean_list):
            sample = tio.Subject(
                dirty=tio.ScalarImage(dirty_path, reader=ALMAreader),
                clean=tio.ScalarImage(clean_path, reader=ALMAreader),
                tclean=tio.ScalarImage(tclean_path, reader=ALMAreader),
            )
            test_samples.append(sample)
    else:
        test_dir = root_dir + 'Test'
        test_dirty_list = np.array(natsorted([
            os.path.join(test_dir, file) for file in os.listdir(test_dir) if 'dirty' in file]))
        test_clean_list = np.array(natsorted([
            os.path.join(test_dir, file) for file in os.listdir(test_dir) if 'clean' in file]))
        test_samples = []
        for (dirty_path, clean_path) in zip(test_dirty_list, test_clean_list):
            sample = tio.Subject(
                dirty=tio.ScalarImage(dirty_path, reader=ALMAreader),
                clean=tio.ScalarImage(clean_path, reader=ALMAreader),
            )
            test_samples.append(sample)
        ids = [int(x.split('_')[-1].split('.')[0]) for x in clean_list]
    
    test_dataset = tio.SubjectsDataset(test_samples, transform=test_transforms)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    ids = np.array(np.array_split(np.array(ids), len(test_dataloader)))
    return test_dataloader, ids
    
def find_params(data_dir, filenames, input_shape, channel_names, preprocess=None,):
    print('Finding normalization parameters...')
    means = []
    stds = []
    maxes = []
    if len(input_shape) == 2:
        for channel in tqdm(channel_names):
            input_names = [filename for filename in filenames if "_".join(filename.split('.')[0].split('_')[2:]) in channel]
            channel_means, channel_stds = [], [] 
            for filename in tqdm(input_names):
                inputs = load_fits(os.path.join(data_dir, filename))
                if preprocess is not None:
                    if preprocess == 'log':
                        inputs = np.log10(inputs + 1e-6)
                    elif preprocess == 'sqrt':
                        inputs = np.sqrt(inputs)
                    elif preprocess == 'log_sqrt':
                        inputs = np.sqrt(np.log10(inputs + 1e-6))
                    elif preprocess == 'sinh':
                        inputs = np.sinh(inputs)
                channel_means.append(inputs.mean())
                channel_stds.append(inputs.std())
                maxes.append(inputs.max())
            means.append(np.mean(channel_means))
            stds.append(np.mean(channel_stds))

        means = np.array(means)
        stds = np.array(stds)
        maxes = np.array(maxes)
        max_ = np.max(maxes)
        print('Finished')
        print('Means: ', means)
        print('Stds: ', stds)
        print('Max: ', max_)
        return means, stds, max_
    elif len(input_shape) == 3:
        return


def TNG_build_dataset(data_path, catalogue_path, preprocess='log',
                      normalize=True,
                      normalize_output=False,
                      channel_names=['Euclid_H', 'Euclid_J',  'Euclid_Y', 'LSST_g', 
                                     'LSST_i', 'LSST_r', 'LSST_u', 'LSST_y', 'LSST_z'], 
                      input_shape=(256, 256), 
                      map='stellarmass', 
                      mode='Train', 
                      batch_size=32,
                      num_workers=8):
    print('Preprocessing data')
    catalogue, filelist, image_ids, file_ids, \
            train_ids, val_ids, test_ids = TNG_prepare_data(data_path, catalogue_path)
    if normalize is True and mode == 'Train':
        mean, std, max = find_params(data_path, filelist, input_shape, channel_names)
        t = A.Compose([
                A.Resize(256, 256),
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.RandomRotate90(p=1), 
                A.Normalize(mean=mean, std=std, max_pixel_value=max),
                ToTensorV2()])
    elif normalize is False and mode == 'Train':
        t = A.Compose([
                A.Resize(256, 256),
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.RandomRotate90(p=1), 
                ToTensorV2()])
    elif normalize is True and mode == 'Valid':
        t = A.Compose([A.Resize(256, 256), A.Normalize(mean=mean, std=std, max_pixel_value=max),
                ToTensorV2()])
    else:
        t = A.Compose([A.Resize(256, 256), ToTensorV2()])
    
    if mode == 'Train':
        data = TNGDataset(data_path, catalogue, filelist, image_ids, file_ids, train_ids, 
                        t, normalize_output, preprocess,
                        channel_names, map, show_plots=False)
        loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        return loader
    elif mode == 'Valid':
        data = TNGDataset(data_path, catalogue, filelist, image_ids, file_ids, val_ids, 
                        t, normalize_output, preprocess,
                        channel_names, map, show_plots=False)
        loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        return loader
    else:
        data = TNGDataset(data_path, catalogue, filelist, image_ids, file_ids, test_ids, 
                        t, normalize_output, preprocess,
                        channel_names, map, show_plots=False)
        loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        return loader

def param_selector(y, config):
    if config['parameter'] == 'x':
        return torch.from_numpy(y[:, 0][:, np.newaxis])
    elif config['parameter'] == 'y':
        return torch.from_numpy(y[:, 1][:, np.newaxis])
    elif config['parameter'] == 'fwhm_x':
        return torch.from_numpy(y[:, 2][:, np.newaxis])
    elif config['parameter'] == 'fwhm_y':
        return torch.from_numpy(y[:, 3][:, np.newaxis])
    elif config['parameter'] == 'pa':
        return torch.from_numpy(y[:, 4][:, np.newaxis])
    elif config['parameter'] == 'flux':
        if config['preprocess'] == 'log':
            return torch.from_numpy(np.log10(y[:, 5] + 1e-6)[:, np.newaxis])
        elif config['preprocess'] == 'sqrt':
            return torch.from_numpy(np.sqrt(y[:, 5])[:, np.newaxis])
        elif config['preprocess'] == 'log_sqrt':
            return torch.from_numpy(np.sqrt(np.log10(y[:, 5] + 1e-6))[:, np.newaxis])
        elif config['preprocess'] == 'sinh':
            return torch.from_numpy(np.sinh(y[:, 5])[:, np.newaxis])
        else:
            return torch.from_numpy(y[:, 5][:, np.newaxis])
    elif config['parameter'] == 'continuum':
        return torch.from_numpy(y[:, 6][:, np.newaxis])

def create_tclean_comparison_images(tclean, inputs, outputs, targets, step, plot_dir, show=False):
    matplotlib.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(32, 32))
    tclean = torch.sum(tclean, dim=4)
    inputs = torch.sum(inputs, dim=4)
    outputs = torch.sum(outputs, dim=4)
    targets = torch.sum(targets, dim=4)
    for i in range(4):
        tcimage = tclean[i, 0].cpu().detach().numpy()
        tcimage = (tcimage - np.min(tcimage)) / (np.max(tcimage) - np.min(tcimage))
        dimage = inputs[i, 0].cpu().detach().numpy()
        dimage = (dimage - np.min(dimage)) / (np.max(dimage) - np.min(dimage))
        bimage = outputs[i, 0].cpu().detach().numpy()
        bimage = (bimage - np.min(bimage)) / (np.max(bimage) - np.min(bimage))
        timage = targets[i, 0].cpu().detach().numpy()
        timage = (timage - np.min(timage)) / (np.max(timage) - np.min(timage))
        ax[i, 0].imshow(dimage, origin='lower', cmap='viridis')
        ax[i, 0].set_xlabel('')
        ax[i, 0].set_ylabel('')
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])
        ax[i, 0].set_title('Dirty Image')
        ax[i, 1].imshow(timage, origin='lower', cmap='viridis')
        ax[i, 1].set_xlabel('')
        ax[i, 1].set_ylabel('')
        ax[i, 1].set_xticks([])
        ax[i, 1].set_yticks([])
        ax[i, 1].set_title('Target Sky Model')
        ax[i, 2].imshow(bimage, origin='lower', cmap='viridis')
        ax[i, 2].set_xlabel('')
        ax[i, 2].set_ylabel('')
        ax[i, 2].set_xticks([])
        ax[i, 2].set_yticks([])
        ax[i, 2].set_title('DeepFocus Prediction')
        ax[i, 3].imshow(tcimage, origin='lower', cmap='viridis')
        ax[i, 3].set_xlabel('')
        ax[i, 3].set_ylabel('')
        ax[i, 3].set_xticks([])
        ax[i, 3].set_yticks([])
        ax[i, 3].set_title('tCLEAN Cleaned Image')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'tclean_comparison_' + str(step) + '.png'))
    if show == True:
        plt.show()
    plt.close() 
    tcimage = tclean[0, 0].cpu().detach().numpy()
    tcimage = (tcimage - np.min(tcimage)) / (np.max(tcimage) - np.min(tcimage))
    timage = targets[0, 0].cpu().detach().numpy()
    timage = (timage - np.min(timage)) / (np.max(timage) - np.min(timage))
    bimage = outputs[0, 0].cpu().detach().numpy()
    bimage = (bimage - np.min(bimage)) / (np.max(bimage) - np.min(bimage))
    #bimage = (bimage - np.min(bimage)) / (np.max(bimage) - np.min(bimage))
    tcres = tcimage - timage
    bres = bimage - timage
    matplotlib.rcParams.update({'font.size': 12})
    plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1,3], height_ratios=[3,1])
    ax = plt.subplot(gs[0,1])
    axl = plt.subplot(gs[0,0])
    axb = plt.subplot(gs[1,1])
    im0 = ax.imshow(bres, origin='lower', aspect='auto')
    plt.colorbar(im0, ax=ax)
    ax.set_title('DeepFocus Residual Image')
    axl.hist(np.sum(bres, axis=1), orientation="horizontal", bins=30, edgecolor='black')
    axb.hist(np.sum(bres, axis=0), bins=30, edgecolor='black')
    axl.set_xticks([])
    axb.set_yticks([])
    axb.set_xlabel('x Residuals')
    axl.set_ylabel('y Residuals')
    plt.savefig(os.path.join(plot_dir, 'deepfocus_residuals_' + str(step) + '.png'))
    if show == True:
        plt.show()
    plt.close()

    plt.figure(figsize=(8, 8))            
    gs = gridspec.GridSpec(2, 2, width_ratios=[1,3], height_ratios=[3,1])
    ax = plt.subplot(gs[0,1])
    axl = plt.subplot(gs[0,0])
    axb = plt.subplot(gs[1,1])
    im1 = ax.imshow(tcres, origin='lower', aspect='auto')
    plt.colorbar(im1, ax=ax)
    ax.set_title('tCLEAN Residual Image')
    axl.hist(np.sum(tcres, axis=1), orientation="horizontal", bins=30, edgecolor='black')
    axb.hist(np.sum(tcres, axis=0), bins=30, edgecolor='black')
    axl.set_xticks([])
    axb.set_yticks([])
    axb.set_xlabel('x Residuals')
    axl.set_ylabel('y Residuals')
    plt.savefig(os.path.join(plot_dir, 'tclean_residuals_' + str(step) + '.png'))
    if show == True:
        plt.show()
    plt.close()

def save_images(outputs, ids, config):
    for i, id in enumerate(ids):
        path = os.path.join(config['prediction_path'], config['name'], 'prediction_' + str(id) + '.fits')
        print('Saving Prediction to ' + path)
        output = outputs[i].cpu().detach().numpy().transpose(0, 3, 1, 2)
        hdu = fits.PrimaryHDU(output)
        hdul = fits.HDUList([hdu])
        hdul.writeto(path, overwrite=True)
        hdul.close()

def log_images(inputs, predictions, targets, step, mode):
    idxs = random.sample(list(np.arange(0, inputs.shape[0])), 2)
    input_log = inputs[idxs]
    prediction_log = predictions[idxs]
    target_log = targets[idxs]
    if len(inputs.shape) == 5:
        input_log = torch.sum(input_log, dim=4)
        prediction_log = torch.sum(prediction_log, dim=4)
        target_log = torch.sum(target_log, dim=4)
    images = torch.cat([input_log, prediction_log, target_log], dim=0)
    images = make_grid(images, nrow=2, normalize=True)
    imgs = wandb.Image(images, caption=f"{mode} Step {step} Top row: inputs, Middle row: predictions, Bottom row: targets")
    if mode == 'Train':
        wandb.log({"Train Images": imgs})
    elif mode == 'Valid':
        wandb.log({"Validation Images": imgs})
    else:
        wandb.log({"Test Images": imgs})

def log_parameters(outputs, targets, step, config, mode):
    data = [[x, y] for (x, y) in zip(targets[:, 0].cpu().detach().numpy(), outputs[:, 0].cpu().detach().numpy())]
    data = wandb.Table(data=data, columns = ["Target", "Prediction"])
    if mode=='Train':
        wandb.log({'Training_plot': wandb.plot.scatter(data, x='Target', 
                            y='Prediction',
                            title='Training Scatter Plot Step: {}'.format(step))})
    
    else:
        wandb.log({'Validation_plot':  wandb.plot.scatter(data, x='Target', 
                            y='Prediction',
                            title='Validation Scatter Plot Step: {}'.format(step))})

def train_batch(inputs, targets, model, optimizer, criterion):
    outputs = model(inputs)
    if isinstance(criterion, list):
        loss = 0
        for criterion_ in criterion:
            loss += criterion_(outputs, targets)
    else:
        loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, outputs

def test_batch(inputs, targets, model, criterion):
    outputs = model(inputs)
    if isinstance(criterion, list):
        loss = 0
        for criterion_ in criterion:
            loss += criterion_(outputs, targets)
    else:
        loss = criterion(outputs, targets)
    return loss, outputs

def test_tclean(tclean, targets, criterion):
    if isinstance(criterion, list):
        loss = 0
        for criterion_ in criterion:
            loss += criterion_(tclean, targets)
    else:
        loss = criterion(tclean, targets)
    return loss

def train_epoch(train_loader, model, optimizer, criterion, config, epoch, example_ct, device):
    model.train()
    nb = len(train_loader)
    running_loss = 0.0
    for i_batch, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        if config['dataset'] == 'TNG':   
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
        else:
            if config['dmode'] == 'deconvolver':
                inputs = batch['dirty'][tio.DATA].to(device)
                targets = batch['clean'][tio.DATA].to(device)
            elif config['dmode'] == 'regressor':
                    inputs = batch[0].to(device)
                    targets = batch[1]
                    targets = param_selector(targets, config).to(device)
        loss, outputs = train_batch(inputs, targets, model, optimizer, criterion)
        example_ct += len(inputs)
        if ((example_ct + 1) % config['log_rate']) == 0:
            wandb.log({'loss': loss.item()})
            if config['dmode'] == 'deconvolver':
                log_images(inputs, outputs, targets, epoch, 'Validation')
            elif config['dmode'] == 'regressor':
                log_parameters(outputs, targets, epoch, config, 'Validation')
        ni = i_batch + nb * epoch
        if ni < config['warm_start_iterations'] and config['warm_start'] is True:
            xi = [0, config['warm_start_iterations']]
            for _, x in enumerate(optimizer.param_groups):
                x['lr'] = np.interp(ni, xi, [0, config['learning_rate']])        
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def valid_epoch(valid_loader, model, criterion, config, epoch, device):
    torch.cuda.empty_cache()
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i_batch, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            if config['dataset'] == 'TNG':   
                inputs = batch[0].to(device)
                targets = batch[1].to(device)
            else:
                if config['dmode'] == 'deconvolver':
                    inputs = batch['dirty'][tio.DATA].to(device)
                    targets= batch['clean'][tio.DATA].to(device)
                elif config['dmode'] == 'regressor':
                    inputs = batch[0].to(device)
                    targets = batch[1]
                    targets = param_selector(targets, config).to(device)
            loss, outputs = test_batch(inputs, targets, model, criterion)
            if ((i_batch + 1) % config['log_rate']) == 0:
                wandb.log({'vloss': loss.item()})
            running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(valid_loader.dataset)
    if config['dmode'] == 'deconvolver':
        log_images(inputs, outputs, targets, epoch, 'Validation')
    elif config['dmode'] == 'regressor':
        log_parameters(outputs, targets, epoch, config, 'Validation')
    return epoch_loss

def test_epoch(test_loader, model, criterion, config, ids, epoch, device):
    torch.cuda.empty_cache()
    model.eval()
    running_loss = 0.0
    model_residuals = []
    
    if config['get_tclean'] is True:
        tclean_running_loss = 0.0
        tclean_residuals = []
    with torch.no_grad():
        for i_batch, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            if config['dataset'] == 'TNG':   
                inputs = batch[0].to(device)
                targets = batch[1].to(device)
            else:
                if config['dmode'] == 'deconvolver':
                    inputs = batch['dirty'][tio.DATA].to(device)
                    targets= batch['clean'][tio.DATA].to(device)
                    if config['get_tclean'] is True:
                        tclean = batch['tclean'][tio.DATA].to(device)
                elif config['dmode'] == 'regressor':
                    inputs = batch[0].to(device)
                    targets = batch[1]
                    targets = param_selector(targets, config).to(device)
            loss, outputs = test_batch(inputs, targets, model, criterion)
            if config['save_predictions'] is True:
                if config['dmode'] == 'deconvolver':
                    save_images(outputs,  ids[i_batch], config)
            if ((i_batch + 1) % config['log_rate']) == 0:
                wandb.log({'test_loss': loss.item()})
            running_loss += loss.item() * inputs.size(0)
            if config['get_tclean'] is True:
                tclean_loss = test_tclean(tclean, targets, criterion)
                if ((i_batch + 1) % config['log_rate']) == 0:
                    wandb.log({'tclean_loss': tclean_loss.item()})
                tclean_running_loss += tclean_loss.item() * inputs.size(0)
            for i in range(inputs.size(0)):
                model_residuals.append(np.mean(targets[i, 0].cpu().detach().numpy() - outputs[i, 0].cpu().detach().numpy()))
                if config['get_tclean'] is True:
                    tclean_residuals.append(np.mean(tclean[i, 0].cpu().detach().numpy() - outputs[i, 0].cpu().detach().numpy()))
            if config['get_tclean'] is True:
                create_tclean_comparison_images(tclean, inputs, outputs, targets, i_batch, os.path.join(config['plot_path'], config['name']))
    epoch_loss = running_loss / len(test_loader.dataset)
    if config['get_tclean'] is True:
        tclean_epoch_loss = tclean_running_loss / len(test_loader.dataset)
        tclean_res = np.mean(tclean_residuals)
    log_images(inputs, outputs, targets, epoch, 'Test')
    model_res = np.mean(model_residuals)
    if config['get_tclean'] is True:
        return epoch_loss, tclean_epoch_loss, model_res, tclean_res
    else:
        return epoch_loss, model_res
                   
def train_sweep(config=None):
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        # Get the current time
        now = datetime.datetime.now()
        # Format the time as a string in the desired format
        time_string = now.strftime("%Y-%m-%d-%H:%M:%S")         
        config['model_name'] = config['dataset'] + '_' + config['dmode'] + '_' + time_string
        print('Model Name: ', config['model_name'])
        if config['dataset'] == 'TNG':
            train_loader = TNG_build_dataset(config['data_path'],
                                             config['catalogue_path'],
                                             preprocess=config['preprocess'],
                                             normalize=config['normalize'],
                                             normalize_output=config['normalize_output'],
                                             channel_names=config['channel_names'],
                                             input_shape=config['input_shape'],
                                             map=config['map_name'], 
                                             mode='Train',
                                             batch_size=config['batch_size'],
                                             num_workers=config['num_workers'])
            valid_loader = TNG_build_dataset(config['data_path'],
                                             config['catalogue_path'],
                                             preprocess=config['preprocess'],
                                             normalize=config['normalize'],
                                             normalize_output=config['normalize_output'],
                                             channel_names=config['channel_names'],
                                             input_shape=config['input_shape'],
                                             map=config['map_name'], 
                                             mode='Valid',
                                             batch_size=config['batch_size'], 
                                             num_workers=config['num_workers'])
        elif config['dataset'] == 'ALMA':
            print('Loading data... for {} model in mode {}'.format(config['model_name'], config['dmode']))
            if config['dmode'] == 'deconvolver':
                if config['normalize'] is True:
                    training_transforms = tio.Compose([
                        tio.RandomFlip(axes=(0, 1), p=1),
                        tio.RandomAffine(scales=0, degrees=(0, 0, 90), translation=0, p=1),
                        tio.CropOrPad((256, 256, 128)),
                        tio.RescaleIntensity((0, 1))
                         ])
                    validation_transforms = tio.Compose([
                        tio.CropOrPad((256, 256, 128)),
                        tio.RescaleIntensity((0, 1))])
                else:
                    training_transforms = tio.Compose([
                        tio.RandomFlip(axes=(0, 1), p=1),
                        tio.RandomAffine(scales=0, degrees=(0, 0, 90), translation=0, p=1),
                        tio.CropOrPad((256, 256, 128))
                        ])
                    validation_transforms = tio.Compose([
                        tio.CropOrPad((256, 256, 128))])
                train_loader, valid_loader = get_ALMA_dataloaders(config['data_path'], 
                                            training_transforms,
                                            validation_transforms,
                                            config['batch_size'],
                                            config['num_workers'])
            elif config['dmode'] == 'regressor':
                train_path = config['data_path'] + 'Train/'
                valid_path = config['data_path'] + 'Validation/'
                train_dataset = ALMADataset( 'train_params.csv', train_path)
                valid_dataset = ALMADataset( 'valid_params.csv', valid_path)
                train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True, collate_fn=train_dataset.collate_fn)
                valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True, collate_fn=valid_dataset.collate_fn)
        print('Data loaded')
        print(config['block'])
        if config['block'] == 'basic':
            encoder_block = ResNetBasicBlock
            decoder_block = ResNetBasicBlock
        elif config['block'] == 'bottleneck':
            encoder_block = ResNetBottleNeckBlock
            decoder_block = ResNetBottleNeckBlock
        else:
            encoder_block = ResNetBasicBlock
            decoder_block = ResNetBasicBlock
            
        model = DeepFocus(in_channels=config['in_channels'], out_channels=config['out_channels'], 
                 blocks_sizes=config['block_sizes'],
                 oblocks_sizes=config['oblock_sizes'],
                 encoder_kernel_sizes=config['kernel_sizes'],
                 depths = config['depths'],
                 decoder_kernel_sizes=config['kernel_sizes'],
                 output_kernel_sizes=config['output_kernel_sizes'],
                 hidden_size=config['hidden_size'],
                 encoder_activation=config['encoder_activation'],
                 decoder_activation=config['decoder_activation'],
                 encoder_block=encoder_block,
                 decoder_block=decoder_block,
                 final_activation=config['final_activation'],
                 input_shape=config['input_shape'],
                 skip_connections=config['skip_connections'],
                 debug=config['debug'],
                 dmode=config['dmode'],
                 dropout_rate=config['dropout_rate'])
        print('Model loaded')
        print('Detected {} GPUs'.format(torch.cuda.device_count()))
        model.to(device)
        print('Saving Model and loading on wandb')
        outpath = os.sep.join((config['output_path'], config['project_name'] + '_' + config['model_name'] + ".h5"))
        woutpath = os.sep.join((config['output_path'], config['project_name'] + '_' + config['model_name'] + ".onnx"))
        if os.path.exists(config['output_path']) is False:
            os.mkdir(config['output_path'])
        if os.path.exists(outpath) == True and config['resume'] == True:
            print('Loading model from {}'.format(outpath))
            model.load_state_dict(torch.load(outpath))
        #torch.save(model.state_dict(), outpath)
        torch.onnx.export(model, 
                        torch.randn(config['input_shape']).unsqueeze(0).unsqueeze(0).to(device),
                        woutpath)
        wandb.save(woutpath, policy='now')

        criterion = build_loss(config['criterion'], config['input_shape'])
        optimizer = build_optimizer(config['optimizer'], 
                                    model, config['learning_rate'],
                                    config['weight_decay'])

        wandb.watch(model, criterion, log='all', log_freq=10)
        example_ct = 0
        best_loss = 99999
        if os.path.exists(config['output_path']) == False:
            os.mkdir(config['output_path'])
        for epoch in tqdm(range(config.epochs), total=config.epochs):
            avg_loss = train_epoch(train_loader, model, optimizer, 
                                    criterion, config, epoch, example_ct, device)
            wandb.log({"train loss": avg_loss, "epoch": epoch})  
            torch.cuda.empty_cache()
            loss = valid_epoch(valid_loader, model, criterion, config, epoch, device)
            wandb.log({"valid loss": loss, "epoch": epoch})
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), outpath)
                torch.onnx.export(model, 
                        torch.randn(config['input_shape']).unsqueeze(0).unsqueeze(0).to(device),
                        woutpath)
                wandb.save(woutpath, policy='now')

def train(config=None):
    with wandb.init(config=config, project=config['project'], 
                    name=config['name'], entity=config['entity']):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        if config['dataset'] == 'TNG':
            train_loader = TNG_build_dataset(config['data_path'],
                                             config['catalogue_path'],
                                             preprocess=config['preprocess'],
                                             normalize=config['normalize'],
                                             normalize_output=config['normalize_output'],
                                             channel_names=config['channel_names'],
                                             input_shape=config['input_shape'],
                                             map=config['map_name'], 
                                             mode='Train',
                                             batch_size=config['batch_size'],
                                             num_workers=config['num_workers'])
            valid_loader = TNG_build_dataset(config['data_path'],
                                             config['catalogue_path'],
                                             preprocess=config['preprocess'],
                                             normalize=config['normalize'],
                                             normalize_output=config['normalize_output'],
                                             channel_names=config['channel_names'],
                                             input_shape=config['input_shape'],
                                             map=config['map_name'], 
                                             mode='Valid',
                                             batch_size=config['batch_size'], 
                                             num_workers=config['num_workers'])
        elif config['dataset'] == 'ALMA':
            print('Loading data... for {} model'.format(config['dmode']))
            if config['dmode'] == 'deconvolver':
                if config['normalize'] is True:
                    training_transforms = tio.Compose([
                        tio.RandomFlip(axes=(0, 1), p=1),
                        tio.RandomAffine(scales=0, degrees=(0, 0, 90), translation=0, p=1),
                        tio.CropOrPad((256, 256, 128)),
                        tio.RescaleIntensity((0, 1))
                         ])
                    validation_transforms = tio.Compose([
                        tio.CropOrPad((256, 256, 128)),
                        tio.RescaleIntensity((0, 1))])
                else:
                    training_transforms = tio.Compose([
                        tio.RandomFlip(axes=(0, 1), p=1),
                        tio.RandomAffine(scales=0, degrees=(0, 0, 90), translation=0, p=1),
                        tio.CropOrPad((256, 256, 128))
                        ])
                    validation_transforms = tio.Compose([
                        tio.CropOrPad((256, 256, 128))])
                train_loader, valid_loader = get_ALMA_dataloaders(config['data_path'], 
                                            training_transforms,
                                            validation_transforms,
                                            config['batch_size'],
                                            config['num_workers'])
            elif config['dmode'] == 'regressor':
                train_path = config['data_path'] + 'Train/'
                valid_path = config['data_path'] + 'Validation/'
                train_dataset = ALMADataset( 'train_params.csv', train_path)
                valid_dataset = ALMADataset( 'valid_params.csv', valid_path)
                train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True, collate_fn=train_dataset.collate_fn)
                valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True, collate_fn=valid_dataset.collate_fn)

        if config['block'] == 'basic':
            encoder_block = ResNetBasicBlock
            decoder_block = ResNetBasicBlock
        elif config['block'] == 'bottleneck':
            encoder_block = ResNetBottleNeckBlock
            decoder_block = ResNetBottleNeckBlock
        else:
            encoder_block = ResNetBasicBlock
            decoder_block = ResNetBasicBlock
            
        model = DeepFocus(in_channels=config['in_channels'], out_channels=config['out_channels'], 
                 blocks_sizes=config['block_sizes'],
                 oblocks_sizes=config['oblock_sizes'],
                 encoder_kernel_sizes=config['kernel_sizes'],
                 depths = config['depths'],
                 decoder_kernel_sizes=config['kernel_sizes'],
                 output_kernel_sizes=config['output_kernel_sizes'],
                 hidden_size=config['hidden_size'],
                 encoder_activation=config['encoder_activation'],
                 decoder_activation=config['decoder_activation'],
                 encoder_block=encoder_block,
                 decoder_block=decoder_block,
                 final_activation=config['final_activation'],
                 input_shape=config['input_shape'],
                 skip_connections=config['skip_connections'],
                 debug=config['debug'],
                 dmode=config['dmode'],
                 dropout_rate=config['dropout_rate'])
        print('Detected {} GPUs'.format(torch.cuda.device_count()))
        model.to(device)
        print('Saving Model and loading on wandb')
        outpath = os.sep.join((config['output_path'], config['project'] + '_' + config['name'] + ".h5"))
        woutpath = os.sep.join((config['output_path'], config['project'] + '_' + config['name'] + ".onnx"))
        if os.path.exists(config['output_path']) is False:
            os.mkdir(config['output_path'])
        if os.path.exists(outpath) == True and config['resume'] == True:
            print('Loading model from {}'.format(outpath))
            model.load_state_dict(torch.load(outpath))
        #torch.save(model.state_dict(), outpath)
        torch.onnx.export(model, 
                        torch.randn(config['input_shape']).unsqueeze(0).unsqueeze(0).to(device),
                        woutpath)
        wandb.save(woutpath, policy='now')

        criterion = build_loss(config['criterion'], config['input_shape'])
        optimizer = build_optimizer(config['optimizer'], 
                                    model, config['learning_rate'],
                                    config['weight_decay'])

        wandb.watch(model, criterion, log='all', log_freq=10)
        example_ct = 0
        best_loss = 99999
        if os.path.exists(config['output_path']) == False:
            os.mkdir(config['output_path'])
        for epoch in tqdm(range(config.epochs), total=config.epochs):
            avg_loss = train_epoch(train_loader, model, optimizer, 
                                    criterion, config, epoch, example_ct, device)
            wandb.log({"train loss": avg_loss, "epoch": epoch})  
            torch.cuda.empty_cache()
            loss = valid_epoch(valid_loader, model, criterion, config, epoch, device)
            wandb.log({"valid loss": loss, "epoch": epoch})
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), outpath)
                torch.onnx.export(model, 
                        torch.randn(config['input_shape']).unsqueeze(0).unsqueeze(0).to(device),
                        woutpath)
                wandb.save(woutpath, policy='now')

def test(config=None):
    with wandb.init(config=config, project=config['project'], name=config['name'] + '_test', entity=config['entity']):
        config = wandb.config
        if os.path.exists(config['prediction_path']) is False:
            os.makedirs(config['prediction_path'])
        if os.path.exists(os.path.join(config['prediction_path'], config['name'])) is False:
            os.makedirs(os.path.join(config['prediction_path'], config['name']))
        print('Loading data... for {} model'.format(config['dmode']))
        if config['normalize']:
            test_transforms = tio.Compose([
                                tio.CropOrPad((256, 256, 128)),
                                tio.RescaleIntensity((0, 1))])
        else:
            test_transforms = tio.Compose([
                                tio.CropOrPad((256, 256, 128))])
        test_dataloader, ids = get_ALMA_test_dataloaders(config['data_path'], config['tclean_path'], 
                                                    test_transforms=test_transforms, 
                                                    batch_size=config['batch_size'],
                                                    num_workers=config['num_workers'], 
                                                    get_tclean=config['get_tclean'])
        if config['block'] == 'basic':
            encoder_block = ResNetBasicBlock
            decoder_block = ResNetBasicBlock
        elif config['block'] == 'bottleneck':
            encoder_block = ResNetBottleNeckBlock
            decoder_block = ResNetBottleNeckBlock
        else:
            encoder_block = ResNetBasicBlock
            decoder_block = ResNetBasicBlock
            
        model = DeepFocus(in_channels=config['in_channels'], out_channels=config['out_channels'], 
                 blocks_sizes=config['block_sizes'],
                 oblocks_sizes=config['oblock_sizes'],
                 encoder_kernel_sizes=config['kernel_sizes'],
                 depths = config['depths'],
                 decoder_kernel_sizes=config['kernel_sizes'],
                 output_kernel_sizes=config['output_kernel_sizes'],
                 hidden_size=config['hidden_size'],
                 encoder_activation=config['encoder_activation'],
                 decoder_activation=config['decoder_activation'],
                 encoder_block=encoder_block,
                 decoder_block=decoder_block,
                 final_activation=config['final_activation'],
                 input_shape=config['input_shape'],
                 skip_connections=config['skip_connections'],
                 debug=config['debug'],
                 dmode=config['dmode'],
                 dropout_rate=config['dropout_rate'])
        print('Detected {} GPUs'.format(torch.cuda.device_count()))
        model.to(device)
        print('Loading Model and loading on wandb')
        outpath = os.sep.join((config['output_path'], config['project'] + '_' + config['name'] + ".h5"))
        woutpath = os.sep.join((config['output_path'], config['project'] + '_' + config['name'] + ".onnx"))
        if os.path.exists(outpath) == True:
            print('Loading model from {}'.format(outpath))
            model.load_state_dict(torch.load(outpath))
        #torch.save(model.state_dict(), outpath)
        #torch.onnx.export(model, 
        #                torch.randn(config['input_shape']).unsqueeze(0).unsqueeze(0).to(device),
        #                woutpath)
        #wandb.save(woutpath, policy='now')
        criterion = build_loss(config['criterion'], config['input_shape'])
        wandb.watch(model, criterion, log='all', log_freq=10)
        if os.path.exists(config['plot_path']) == False:
            os.mkdir(config['plot_path'])
        if os.path.exists(os.path.join(config['plot_path'], config['name'])) is False:
            os.mkdir(os.path.join(config['plot_path'], config['name']))
        if config['get_tclean'] == True:
            loss, tclean_loss, res, tclean_res = test_epoch(test_dataloader, model, criterion, config, ids, epoch=0, device=device)
            print('DeepFocus Loss: {}, Tclean Loss: {}'.format(loss, tclean_loss))
            print('DeepFocus Residual: {}, Tclean Residual: {}'.format(res, tclean_res))
            wandb.log({"test loss": loss, "test tclean loss": tclean_loss, "test residual": res, "test tclean residual": tclean_res})
        else:
            loss, res = test_epoch(test_dataloader, model, criterion, config, ids, epoch=0, device=device)
            print('DeepFocus Loss: {}}'.format(loss))
            print('DeepFocus Residual: {}'.format(res))
            wandb.log({"test loss": loss, "test residual": res}) 