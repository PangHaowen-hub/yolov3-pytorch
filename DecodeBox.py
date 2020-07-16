# from __future__ import division  # 将新版本的特性引进当前版本中，division精确除法
import torch
import torch.nn as nn


class BoxDecode(nn.Module):
    def __init__(self, anchors, num_classes, img_size):  # 某一个尺度anchors，其中包含三个anchor,
        super(BoxDecode, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)  # 3个anchor
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes  # 每个anchor有多少个属性即x,y,w,h,置信度,类别20
        self.img_size = img_size

    def forward(self, input):  # 此处input为batch_size * 75=(20+5)*3 * 长 * 宽
        batch_size = input.size(0)  # input的第0维为batch_size
        input_height = input.size(2)  # feature map长宽
        input_width = input.size(3)

        # 计算感受野
        stride_h = self.img_size[1] / input_height  # 图片大小除以feature map大小即为一个anchor对应的原图的大小
        stride_w = self.img_size[0] / input_width
        # 归一到特征层上
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                          self.anchors]  # config中的anchor大小都是指原图中的大小，将其除以感受野大小，得到其在feature map上的大小

        # 对预测结果进行resize
        prediction = input.view(batch_size, self.num_anchors,
                                self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()
        # permute(0, 1, 3, 4, 2)将矩阵按原顺序的01342排列10*3*13*13*25  contiguous()把tensor变成在内存中连续分布的形式
        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])  # prediction中最后一维0-4分别为x,y,w,h,置信度
        y = torch.sigmoid(prediction[..., 1])
        # 先验框的宽高调整参数
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height

        # 获得置信度，是否有物体
        conf = torch.sigmoid(prediction[..., 4])
        # 种类置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # 生成网格，先验框中心，网格左上角
        # torch.linspace(start, end, steps)返回一个一维的tensor，这个张量包含了从start到end，分成steps个线段得到的向量。
        # repeat()沿着指定的维度重复tensor
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_width, 1).repeat(
            batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_height, 1).t().repeat(
            batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)

        # 生成先验框的宽高
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

        # 计算调整后的先验框中心与宽高
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # 用于将输出调整为相对于416x416的大小
        _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,
                            conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
        return output.data
