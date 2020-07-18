import torch
import torch.nn as nn
from darknet import darknet53
from config import Config


def conv2d(channel_in, channel_out, kernel_size):
    pad = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False),
        nn.BatchNorm2d(channel_out),
        nn.LeakyReLU(0.1)
    )


def branch_layers(channels_list, in_channels, final_channel):
    m = nn.ModuleList([  # ModuleList与Sequential的区别见 https://zhuanlan.zhihu.com/p/75206669
        conv2d(in_channels, channels_list[0], 1),
        conv2d(channels_list[0], channels_list[1], 3),
        conv2d(channels_list[1], channels_list[0], 1),
        conv2d(channels_list[0], channels_list[1], 3),
        conv2d(channels_list[1], channels_list[0], 1),
        conv2d(channels_list[0], channels_list[1], 3),
        nn.Conv2d(channels_list[1], final_channel, kernel_size=1, stride=1, padding=0, bias=True)
    ])
    return m


def branch(branch_layer, x):
    for i, e in enumerate(branch_layer):
        x = e(x)
        if i == 4:
            out_branch = x
    return x, out_branch  # x为最终输出，out_branch为和上层合并的feature map


class YoloV3(nn.Module):
    def __init__(self, config):
        super(YoloV3, self).__init__()
        self.config = config

        self.backbone = darknet53(None)

        darknet_out_channels = [256, 512, 1024]  # darknet输出的通道数

        final_channel0 = len(config["yolo"]["anchors"][0]) * (5 + config["yolo"]["classes"])  # 计算最终通道数3*(5+20)=75
        self.branch_layer0 = branch_layers([512, 1024], darknet_out_channels[2], final_channel0)

        final_channel1 = len(config["yolo"]["anchors"][1]) * (5 + config["yolo"]["classes"])
        self.branch_layer1_conv = conv2d(512, 256, 1)
        self.branch_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.branch_layer1 = branch_layers([256, 512], darknet_out_channels[1] + 256, final_channel1)

        final_channel2 = len(config["yolo"]["anchors"][2]) * (5 + config["yolo"]["classes"])
        self.branch_layer2_conv = conv2d(256, 128, 1)
        self.branch_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.branch_layer2 = branch_layers([128, 256], darknet_out_channels[0] + 128, final_channel2)

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)  # x2, x1, x0分别为darknet中out3, out4, out5
        out0, out0_branch = branch(self.branch_layer0, x0)
        x1_in = self.branch_layer1_conv(out0_branch)
        x1_in = self.branch_layer1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = branch(self.branch_layer1, x1_in)

        x2_in = self.branch_layer2_conv(out1_branch)
        x2_in = self.branch_layer2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, _ = branch(self.branch_layer2, x2_in)
        return out0, out1, out2
