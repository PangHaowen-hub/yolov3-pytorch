import torch
import torch.nn as nn
from config import Config
from yolov3 import YoloV3


class BoxDecode(nn.Module):
    def __init__(self, anchors, num_classes, img_size):  # 某一个尺度anchors,其中包含三个anchor,一次传入三个
        super(BoxDecode, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)  # 3个anchor
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes  # 每个anchor有多少个属性即x,y,w,h,置信度,类别20
        self.img_size = img_size

    def forward(self, input):  # 此处input为batch_size * 75=(20+5)*3 * 长 * 宽，即yolov3的输出
        batch_size = input.size(0)  # input的第0维为batch_size
        input_height = input.size(2)  # feature map长
        input_width = input.size(3)  # feature map宽

        # 计算感受野
        stride_h = self.img_size[1] / input_height  # 图片大小除以feature map大小即为感受野大小 416/13=32 416/26=16 416/52=8
        stride_w = self.img_size[0] / input_width
        # config中的anchor大小都是指原图中的大小，将其除以感受野大小，得到其在feature map上的大小
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                          self.anchors]  # [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]
        # 对预测结果进行resize
        prediction = input.view(batch_size, self.num_anchors,
                                self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()
        # permute(0, 1, 3, 4, 2)将矩阵按原顺序的01342排列N*3*13*13*25  contiguous()把tensor变成在内存中连续分布的形式
        # 获取先验框的大小位置、置信度、类别信息
        x = torch.sigmoid(prediction[..., 0])  # x torch.Size([1, 3, 13, 13]) 且做了sigmoid归一化
        y = torch.sigmoid(prediction[..., 1])  # y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        conf = torch.sigmoid(prediction[..., 4])  # 置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])  # 各类别置信度 torch.Size([1, 3, 13, 13, 20])

        float_tensor = torch.cuda.Tensor if x.is_cuda else torch.Tensor

        # 生成网格，先验框中心为网格左上角
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_width, 1).repeat(
            batch_size * self.num_anchors, 1, 1).view(x.shape).type(float_tensor)  # torch.Size([1, 3, 13, 13])
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_height, 1).t().repeat(
            batch_size * self.num_anchors, 1, 1).view(y.shape).type(float_tensor)  # torch.Size([1, 3, 13, 13])
        # 生成先验框的宽高
        anchor_w = float_tensor(scaled_anchors).index_select(1, torch.tensor(0))  # 取scaled_anchors第0列,三个anchor的宽
        anchor_h = float_tensor(scaled_anchors).index_select(1, torch.tensor(1))  # torch.Size([3, 1])
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        # torch.Size([1, 3, 13, 13])
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)
        # 计算调整后的先验框中心与宽高
        pred_boxes = float_tensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # 用于将输出调整为相对于416x416的大小
        scale = torch.tensor([stride_w, stride_h] * 2).type(float_tensor)  # 注意tensor与Tensor区别
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * scale,
                            conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)),
                           -1)  # dim=-1 表示倒数第一维
        return output.data
