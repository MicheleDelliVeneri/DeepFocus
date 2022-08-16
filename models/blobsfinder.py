import torch
import torch.nn as nn
import torch.nn.functional as F 


class batch_act(nn.Module):
    def __init__(self, in_c):
        super(batch_act, self).__init__()
        self.bn = nn.BatchNorm2d(in_c)
        self.act = nn.LeakyReLU()
    def forward(self, inputs):
        x = self.act(inputs)
        x = self.bn(x)
        return x

class residual_block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super(residual_block, self).__init__()
        # Convolutional Layers
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.b1 = batch_act(out_c)
        self.c2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1)
        self.b2 = batch_act(out_c)

        #shortcut connection 
        #self.s = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=stride)
    def forward(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.c2(x)
        x = self.b2(x)
        #s = self.s(inputs)
        #x = x + s
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

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.upsample = Interpolate(scale_factor=2, mode='bilinear', align_corners=True)
        self.c = nn.ConvTranspose2d(in_c, out_c, kernel_size=(3, 3), stride=2, padding=1, output_padding=1)
        self.r = residual_block(in_c + out_c, out_c)
    
    def forward(self, inputs):
        x = self.upsample(inputs)
        x1 = self.c(inputs)
        x = torch.cat([x, x1], dim=1)
        #print(x.shape, skip.shape)
        x = self.r(x)
        return x

class Encoder(nn.Module):
    def __init__(self, hidden_c):
        super(Encoder, self).__init__()
        self.b1 = residual_block(1, 8, stride=2)
        self.b2 = residual_block(8, 16, stride=2)
        self.b3 = residual_block(16, 32, stride=2)
        self.b4 = residual_block(32, 64, stride=2)
        self.dense = nn.Sequential(
            nn.Linear(64 * 16 *16, hidden_c),
            nn.LeakyReLU())
    
    def forward(self, inputs):
        #print(inputs.shape)
        skip1 = self.b1(inputs)
        #print(skip1.shape)
        skip2 = self.b2(skip1)
        #print(skip2.shape)
        skip3 = self.b3(skip2)
        #print(skip3.shape)
        x = self.b4(skip3)
        #print(x.shape)
        x = torch.flatten(x, start_dim=1)
        #print(x.shape)
        x = self.dense(x)
        #print(x.shape)
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_c):
        super(Decoder, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(hidden_c, 64 * 16 * 16),
            nn.LeakyReLU()
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 16, 16))
        self.b1 = decoder_block(64, 32)
        self.b2 = decoder_block(32, 16)
        self.b3 = decoder_block(16, 8)
        self.b4 = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear', align_corners=True),
            residual_block(8, 1))
        self.out = nn.Sequential(
            nn.Conv2d(1,  1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
    def forward(self, inputs):
        #print(inputs.shape)
        x = self.dense(inputs)
        #print(x.shape)
        x = self.unflatten(x)
        #print(x.shape, skip3.shape)
        x = self.b1(x)
        #print(x.shape, skip2.shape)
        x = self.b2(x)
        #print(x.shape, skip1.shape)
        x = self.b3(x)
        #print(x.shape)
        x = self.b4(x)
        #print(x.shape)
        x = self.out(x)
        #print(x.shape)
        return x

class BlobsFinder(nn.Module):
    def __init__(self, hidden_c):
        super(BlobsFinder, self).__init__()
        self.enc = Encoder(hidden_c)
        self.dec = Decoder(hidden_c)
        self.__init__params()
        
    def __init__params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias,   0)

    def forward(self, inputs):
        lat = self.enc(inputs)
        out = self.dec(lat)
        return out