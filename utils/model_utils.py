import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import SSIMLoss
import torch.optim as optim
import wandb
from torchvision.utils import make_grid
import torchio as tio
from torch.autograd import Variable
from functools import partial
import numpy as np
from math import exp


# ----------------- DEEEP LEARNING FUNCTIONS AND CLASSES ----------------- #


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