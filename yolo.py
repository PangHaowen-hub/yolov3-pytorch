import cv2
import numpy as np
import colorsys  # 此模块提供了用于RGB和YIQ/HLS/HSV颜色模式的转换的接口
import os
import torch
import torch.nn as nn
from yolov3 import YoloV3
import torch.backends.cudnn as cudnn
from PIL import Image, ImageFont, ImageDraw
from config import Config
from BoxDecode import BoxDecode
from non_max_suppression import non_max_suppression
from letterbox_image import letterbox_image
from correct_box import yolo_correct_boxes


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/model.pth',
        "classes_path": 'model_data/coco_classes.txt',
        "model_image_size": (416, 416, 3),
        "confidence": 0.5,
        "cuda": True
    }

    @classmethod  # 定义类方法，可以不通过实例来调用类的函数属性，而直接用类调用函数方法
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # 初始化YOLO
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.config = Config
        self.generate()

    # 获取分类名称
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:  # 当with as代码块结束时，自动关闭打开的文件，不会造成系统资源的长期占用
            class_names = f.readlines()  # readlines读取所有行并返回列表
        class_names = [c.strip() for c in class_names]  # strip()用于移除字符串头尾的字符（默认为空格或换行符）
        return class_names

    #
    def generate(self):
        self.config["yolo"]["classes"] = len(self.class_names)
        self.net = YoloV3(self.config)

        # 加快模型训练的效率
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        self.net = self.net.eval()

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(BoxDecode(self.config["yolo"]["anchors"][i], self.config["yolo"]["classes"],
                                               (self.model_image_size[1], self.model_image_size[0])))

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]  # 生成20或80种颜色列表
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))  # 用lambda定义一个简单函数，再结合map使用
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        # 此处先将hsv格式转换为rgb格式，由于默认为0-1小数，然后分别乘255
    # 预测图片，并画出框
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])

        crop_img = np.array(letterbox_image(image, (self.model_image_size[0], self.model_image_size[1])))
        photo = np.array(crop_img, dtype=np.float32)
        photo /= 255.0
        photo = np.transpose(photo, (2, 0, 1))  # 将通道放到最前面
        photo = photo.astype(np.float32)
        images = []
        images.append(photo)

        images = np.asarray(images)
        images = torch.from_numpy(images)
        if self.cuda:
            images = images.cuda()

        with torch.no_grad():
            outputs = self.net(images)
            output_list = []
            for i in range(3):
                output_list.append(self.yolo_decodes[i](outputs[i]))
            output = torch.cat(output_list, 1)  # 将所有先验框放到一个list中，大小为：图片张数 * 10647 * 类别
            batch_detections = non_max_suppression(output, self.config["yolo"]["classes"],
                                                   conf_thres=self.confidence,
                                                   nms_thres=0.3)

        try:
            batch_detections = batch_detections[0].cpu().numpy()  # 此处为了去除最外层括号
        except:
            return image
        # 此时batch_detections中有七列，分别为四个坐标、置信度、类别置信度、类别
        top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
        top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
        top_label = np.array(batch_detections[top_index, -1], np.int32)  # 得出各个类别
        top_bboxes = np.array(batch_detections[top_index, :4])  # 得出各框坐标
        top_xmin = np.expand_dims(top_bboxes[:, 0], -1)
        top_ymin = np.expand_dims(top_bboxes[:, 1], -1)
        top_xmax = np.expand_dims(top_bboxes[:, 2], -1)
        top_ymax = np.expand_dims(top_bboxes[:, 3], -1)

        # 将先验框放缩到原图大小，因为之前将图片加了灰条
        # 此处model_image_size为模型需要的图像尺寸，image_shape为输入图像尺寸
        boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                   np.array([self.model_image_size[0], self.model_image_size[1]]), image_shape)

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0]

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.class_names.index(predicted_class)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image
