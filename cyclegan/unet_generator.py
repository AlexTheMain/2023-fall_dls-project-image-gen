import torch
import torch.nn as nn
    
class ConvOut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.out(x)
    
    
class СonvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()

        self.conv_block= nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_block(x)
    

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv_block = nn.Sequential(
            СonvBlock(in_channels, out_channels),
            СonvBlock(out_channels, out_channels),
        )

    def forward(self, x):
        return self.double_conv_block(x)
    

    
class DownScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2),
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)
    
class UpScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.upsampling_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2,2),
            DoubleConvBlock(out_channels, out_channels),
        )
    
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.upsampling_conv(x)
    

class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.bottleneck_block = nn.Sequential(
            СonvBlock(in_channels, in_channels),
            СonvBlock(in_channels, in_channels),
            nn.ConvTranspose2d(in_channels, out_channels, 2,2),
        )

    def forward(self, x):
        return self.bottleneck_block(x)
    

class UNetGenerator(nn.Module):
    def __init__(self, in_channels = 3, out_channels = None):
        super(UNetGenerator, self).__init__()
        if out_channels==None:
            out_channels=in_channels

        self.input = DoubleConvBlock(in_channels, 32)
        self.ds1 = DownScaleBlock(32,64)
        self.ds2 = DownScaleBlock(64, 128)
        self.ds3 = DownScaleBlock(128, 256)
        self.ds4 = DownScaleBlock(256, 512)
        self.ds5 = DownScaleBlock(512,1024)

        self.bn = BottleNeckBlock(1024, 512)

        self.us1 = UpScaleBlock(1024, 256)
        self.us2 = UpScaleBlock(512, 128)
        self.us3 = UpScaleBlock(256, 64)
        self.us4 = UpScaleBlock(128, 32)
        self.pre_out = DoubleConvBlock(32, 32)
        self.output = ConvOut(32, 3)
    
    def forward(self, x):
        inp = self.input(x)
        x1 = self.ds1(inp)
        x2 = self.ds2(x1)
        x3 = self.ds3(x2)
        x4 = self.ds4(x3)
        x5 = self.ds5(x4)
        
        bn = self.bn(x5)

        x = self.us1(bn, x4)
        x = self.us2(x, x3)
        x = self.us3(x, x2)
        x = self.us4(x, x1)
        x = self.pre_out(x)
        result = self.output(x)

        return result


