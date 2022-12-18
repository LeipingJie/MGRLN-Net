import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UNetConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class UNetModule(nn.Module):
    def __init__(self, n_class, ch_in, ch_base=32, with_output=True):
        super(UNetModule, self).__init__()
        self.with_output = with_output
        chs = [ch_base, ch_base*2, ch_base*4, ch_base*8]
        # encoder
        self.conv_down1 = UNetConvBlock(ch_in, chs[0])
        self.conv_down2 = UNetConvBlock(chs[0], chs[1])
        self.conv_down3 = UNetConvBlock(chs[1], chs[2])
        self.conv_down4 = UNetConvBlock(chs[2], chs[3])

        self.maxpool = nn.MaxPool2d(2)
        
        # decoder
        self.conv_up3 = UNetConvBlock(chs[3] + chs[2], chs[2])
        self.conv_up2 = UNetConvBlock(chs[2] + chs[1], chs[1])
        self.conv_up1 = UNetConvBlock(chs[1] + chs[0], chs[0])
        
        if with_output:
            # output
            self.conv_last = nn.Conv2d(chs[0], n_class, 1)

    def forward(self, x):
        # encoder
        conv1 = self.conv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.conv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.conv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.conv_down4(x)
        
        x = F.interpolate(x, conv3.shape[-2:], mode='bilinear', align_corners=True)      
        x = torch.cat([x, conv3], dim=1)
        
        x = self.conv_up3(x)
        x = F.interpolate(x, conv2.shape[-2:], mode='bilinear', align_corners=True)            
        x = torch.cat([x, conv2], dim=1)       

        x = self.conv_up2(x)
        x = F.interpolate(x, conv1.shape[-2:], mode='bilinear', align_corners=True)            
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.conv_up1(x)
        
        if self.with_output:
            out = self.conv_last(x)
            return out
        
        return x