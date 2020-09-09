from PIL import Image
import numpy as np


# 给图片加边框，防止因拉伸图片使图片失真
def letterbox_image(image, size):  # image为图片矩阵，size为模型要求的图片长宽
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)  # NEAREST低质量；BILINEAR双线性；BICUBIC 三次样条插值；ANTIALIAS高质量
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image
