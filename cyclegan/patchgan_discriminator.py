import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self, res_blocks_length=9):
        super(PatchGANDiscriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),

        )

    def forward(self, x):
        return self.discriminator(x)