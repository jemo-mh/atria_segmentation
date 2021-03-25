
from torch import nn, optim
import torch

n_class =2

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):

    def __init__(self, n_class=n_class):
        super().__init__()

        self.dconv_down1 = double_conv(1, 64)

        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample3 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last1 = nn.Conv2d(64, n_class, 1)
        
    def forward(self, x):
        # x=x.permute(0,3,1,2)
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)
        
        x = self.upsample3(x)
        x = torch.cat([x, conv3], dim=1) 

        x = self.dconv_up3(x) 
        x = self.upsample2(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample1(x)
        x = torch.cat([x, conv1], dim=1)
        
        x = self.dconv_up1(x)
        out1 = self.conv_last1(x)
        
        return out1
    

#_, predicted = torch.max(outputs.data, 1)