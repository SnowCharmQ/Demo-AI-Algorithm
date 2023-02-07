import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=(1, 1)):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.shortcut = nn.Sequential()
        if ch_in != ch_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=(1, 1), stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.block1 = ResNetBlock(64, 128, stride=1)
        self.block2 = ResNetBlock(128, 256, stride=2)
        self.block3 = ResNetBlock(256, 512, stride=2)
        self.block4 = ResNetBlock(512, 512, stride=1)
        self.output = nn.Linear(512 * 2 * 2, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(-1, 512 * 2 * 2)
        x = self.output(x)
        return x