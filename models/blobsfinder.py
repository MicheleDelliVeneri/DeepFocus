import torch
import torch.nn as nn
import torch.nn.functional as F 


# Added this View function to save the model in .oxx
#class View(nn.Module):
#    def __init__(self, dim,  unflattened_size):
#        super(View, self).__init__()
#        self.dim = dim
#        self.shape = unflattened_size
#
#    def forward(self, input):
#        new_shape = list(input.shape)[:self.dim] + list(self.shape) + list(input.shape)[self.dim+1:]
#        return input.view(*new_shape)
#
#
#nn.Unflatten = View

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

class Enc2D(nn.Module):
    def __init__(self, c_in, c_hidden, act_fn):
        super(Enc2D, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.act_fn = act_fn

        self.block1 = nn.Sequential(
            nn.Conv2d(c_in, 8, kernel_size=(3, 3), bias=False, stride=2, padding=1),
            self.act_fn(),
            nn.BatchNorm2d(8)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(3, 3), bias=False, stride=2, padding=1),
            self.act_fn(),
            nn.BatchNorm2d(16)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), bias=False, stride=2, padding=1),
            self.act_fn(),
            nn.BatchNorm2d(32)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), bias=False, stride=2, padding=1),
            self.act_fn(),
            nn.BatchNorm2d(64)
        )

        self.dense = nn.Linear(64 * 16 * 16, self.c_hidden)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense(x)
        return x

class Dec2D(nn.Module):
    def __init__(self, c_in, c_hidden, act_fn):
        super(Dec2D, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.act_fn = act_fn

        self.dense = nn.Sequential(
            nn.Linear(self.c_hidden,  64  * 16 * 16),
            self.act_fn()
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 16, 16))
        self.block1 = nn.Sequential(
            Interpolate(scale_factor=(2, 2), mode='bilinear'),
            nn.ConvTranspose2d(self.c_in, 32, kernel_size=(3, 3), bias=False, stride=1, padding=1),
            self.act_fn(), 
            nn.BatchNorm2d(32)
            )
        self.block2 = nn.Sequential(
            Interpolate(scale_factor=(2, 2), mode='bilinear'),
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), bias=False, stride=1, padding=1),
            self.act_fn(), 
            nn.BatchNorm2d(16)
            )
        self.block3 = nn.Sequential(
            Interpolate(scale_factor=(2, 2), mode='bilinear'),
            nn.ConvTranspose2d(16, 8, kernel_size=(3, 3), bias=False, stride=1, padding=1),
            self.act_fn(), 
            nn.BatchNorm2d(8)
            )
        self.block4 = nn.Sequential(
            Interpolate(scale_factor=(2, 2), mode='bilinear'),
            nn.ConvTranspose2d(8, 1, kernel_size=(3, 3), bias=False, stride=1, padding=1),
            self.act_fn()
            )
        self.out = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.dense(x)
        x = self.unflatten(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.out(x)
        return x

class BlobsFinder(nn.Module):
    def __init__(self, c_in, c_latent, c_out, act_fn):
        super(BlobsFinder, self).__init__()

        self.c_in = c_in
        self.c_latent = c_latent
        self.c_out = c_out
        act_fn_name = {"tanh": nn.Tanh, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "gelu": nn.GELU, 
                       "identity": nn.Identity, "sigmoid": nn.Sigmoid}
        self.act_fn = act_fn_name[act_fn]

        self.enc = Enc2D(self.c_in, self.c_latent, self.act_fn)
        self.dec = Dec2D(self.c_out, self.c_latent, self.act_fn)

        self._init_params(act_fn)
    
    def _init_params(self, act_fn):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if act_fn == "relu" or "leaky_relu":
                    nn.init.kaiming_normal_(m.weight, nonlinearity=act_fn)
                elif act_fn == "tanh":
                    nn.init.xavier_uniform_(
                        m.weight, gain=nn.init.calculate_gain("tanh")
                    )
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        lat = self.enc(x)
        out = self.dec(lat)
        return out