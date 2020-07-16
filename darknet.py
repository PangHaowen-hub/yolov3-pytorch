import torch
import torch.nn as nn
import math
from collections import OrderedDict


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels[1], channels[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += x
        return out


class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.layer0 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(32),
                                    nn.LeakyReLU(0.1))
        self.layer1 = self.make_layer([32, 64], layers[0])  # 1
        self.layer2 = self.make_layer([64, 128], layers[1])  # 2
        self.layer3 = self.make_layer([128, 256], layers[2])  # 8
        self.layer4 = self.make_layer([256, 512], layers[3])  # 8
        self.layer5 = self.make_layer([512, 1024], layers[4])  # 4
        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 用isinstance()判断m是否为nn.Conv2d类
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))  # 正态分布初始化
            elif isinstance(m, nn.BatchNorm2d):  # 用isinstance()判断m是否为nn.BatchNorm2d类
                m.weight.data.fill_(1)  # weight为1
                m.bias.data.zero_()  # bias为0

    def make_layer(self, channels, numbers):  # out_channels为[32, 64]等, numbers为1 2 8 8 4
        layers = []
        # append() 方法用于在列表末尾添加新的对象
        layers.append(("conv", nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(("bn", nn.BatchNorm2d(channels[1])))
        layers.append(("relu", nn.LeakyReLU(0.1)))
        # 加入残差模块
        for i in range(0, numbers):
            layers.append(("residual_{}".format(i), ResidualBlock(channels)))
        return nn.Sequential(OrderedDict(layers))  # 使用OrderedDict会根据放入元素的先后顺序进行排序。所以输出的值是排好序的。

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5


def darknet53(pretrained):
    model = DarkNet([1, 2, 8, 8, 4])
    if pretrained:
        if isinstance(pretrained, str):  # 判断是pretrained否为str
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))

    return model

