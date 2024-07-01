import mindspore.dataset.vision as vision
import numpy as np
from PIL import Image


def image_scale(img, size):
    w, h = img.size
    if (w <= h and w == size) or (h <= w and h == size):
        return img
    if w < h:
        ow = size
        oh = int(size * h / w)
        return img.resize((ow, oh), Image.BILINEAR)
    else:
        oh = size
        ow = int(size * w / h)
    return img.resize((ow, oh), Image.BILINEAR)


def image_random_crop(img, size):
    # crop size should be integer, output a square
    w, h = img.size
    ow, oh = size, size
    i, j = 0, 0
    if w != ow or h != oh:
        i = np.random.randint(0, h - oh + 1, 1)
        j = np.random.randint(0, w - ow + 1, 1)
    return img.crop((i, j, h, w))


def image_random_horizontal_flip(img, p):
    if np.random.rand(1) < p:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def image_normalize(img, mean, std):
    img = np.array(img, dtype='float32') / 255.0
    img = (img - mean) / std
    return img


class ImageTransform:
    def __init__(self, size=256, crop_size=256, mean=(0.5,), std=(0.5,), is_train=True):
        self.size = size
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        self.is_train = is_train

    def __call__(self, img):
        img_scale = image_scale(img, self.size)
        img_crop = image_random_crop(img_scale, self.crop_size)
        img_flip = image_random_horizontal_flip(img_crop, p=0.5) if self.is_train else img_crop
        img_tensor = vision.ToTensor()(img_flip)
        img_normal = vision.Normalize(self.mean, self.std)(img_tensor)
        return img_normal
