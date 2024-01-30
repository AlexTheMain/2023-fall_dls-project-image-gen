# Импорт необходимых модулей и библиотек
import torch
import torch.nn as nn

class InstanseResidualBlock(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, kernel_size=3, stride=1):
        super().__init__()
        # Определение последовательности блоков
        self.res_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=1),
            nn.InstanceNorm2d(out_channels)
        )
        # Использование ReLU в качестве активации
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        input = x 
        x = self.res_block(x)
        # Применение активации к выходу суммы входа и результата блока
        return self.relu_out(x + input)


class ClassicResidualBlock(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, kernel_size=3, stride=1):
        super().__init__()
        # Определение последовательности блоков
        self.res_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        # Использование ReLU в качестве активации
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        input = x
        x = self.res_block(x)
        # Применение активации к выходу суммы входа и результата блока
        return self.relu_out(x + input)


class DownSamplingInput(nn.Module):
    def __init__(self):
        super().__init__()
        # Определение последовательности блоков для уменьшения размерности входных изображений
        self.ds_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.ds_block(x)


class UpSamplingOutput(nn.Module):
    def __init__(self):
        super().__init__()
        # Определение последовательности блоков для увеличения размерности выходных изображений
        self.us_block = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.us_block(x)


class DefaultGenerator(nn.Module):
    def __init__(self, res_blocks_length=9):
        super(DefaultGenerator, self).__init__()
        # Определение последовательности блоков для генератора
        self.ds = DownSamplingInput()
        self.residual_blocks = nn.Sequential(*[InstanseResidualBlock() for _ in range(res_blocks_length)])
        self.us = UpSamplingOutput()
    
    def forward(self, x):
        x = self.ds(x)
        x = self.residual_blocks(x)
        result = self.us(x)
        return result
