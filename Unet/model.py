"""
这是根据UNet模型搭建出的一个基本网络结构
输入和输出大小是一样的，可以根据需求进行修改
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


# 基本卷积块
class Conv(nn.Module):
    def __init__(self, c_in, c_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 1, 1),
            nn.BatchNorm2d(c_out),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(c_out, c_out, 3, 1, 1),
            nn.BatchNorm2d(c_out),
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


# 下采样模块
class DownSampling(nn.Module):
    def __init__(self, c):
        super(DownSampling, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(c, c, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.down(x)


# 上采样模块
class UpSampling(nn.Module):

    def __init__(self, c):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.up = nn.Conv2d(c, c // 2, 1, 1)

    def forward(self, x, r):
        # 使用邻近插值进行下采样
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up(x)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)


# 主干网络
class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        # 4次下采样
        self.c1 = Conv(3, 64)
        self.d1 = DownSampling(64)
        self.c2 = Conv(64, 128)
        self.d2 = DownSampling(128)
        self.c3 = Conv(128, 256)
        self.d3 = DownSampling(256)
        self.c4 = Conv(256, 512)
        self.d4 = DownSampling(512)
        self.c5 = Conv(512, 1024)

        # 4次上采样
        self.u1 = UpSampling(1024)
        self.c6 = Conv(1024, 512)
        self.u2 = UpSampling(512)
        self.c7 = Conv(512, 256)
        self.u3 = UpSampling(256)
        self.c8 = Conv(256, 128)
        self.u4 = UpSampling(128)
        self.c9 = Conv(128, 64)

        self.th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        # 下采样部分
        r1 = self.c1(x)
        r2 = self.c2(self.d1(r1))
        r3 = self.c3(self.d2(r2))
        r4 = self.c4(self.d3(r3))
        y1 = self.c5(self.d4(r4))

        # 上采样部分
        # 上采样的时候需要拼接起来
        o1 = self.c6(self.u1(y1, r4))
        o2 = self.c7(self.u2(o1, r3))
        o3 = self.c8(self.u3(o2, r2))
        o4 = self.c9(self.u4(o3, r1))

        # 输出预测，这里大小跟输入是一致的
        # 可以把下采样时的中间抠出来再进行拼接，这样修改后输出就会更小
        return self.th(self.pred(o4))
