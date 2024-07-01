import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.nn import CrossEntropyLoss, Dropout, Softmax, Dense, Conv2d, LayerNorm
import mindspore.common.initializer as init
from mindspore.common.initializer import initializer
from typing import Any
from libs.fix_weight_dict import fix_model_state_dict
import math
import copy
import numpy as np


if __name__ == '__main__':
    from layers import *
else:
    from .layers import *
def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                initializer(init.Normal(0.0, 0.02), shape=m.shape, dtype=mindspore.float32)
            elif init_type == 'xavier':
                init.XavierNormal(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.HeNormal(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.Orthogonal(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias = init.Constant(0.0)

    return init_fun

class Cvi(nn.Cell):
    def __init__(self, in_channels, out_channels, before=None, after=None, kernel_size=4, stride=2,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Cvi, self).__init__(auto_prefix=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='pad', padding=padding,
                              dilation=dilation, group=groups, has_bias=bias, weight_init=init.Normal(0.0, 0.02))
        if after=='BN':
            self.after = nn.BatchNorm2d(out_channels)
        elif after=='Tanh':
            self.after = ops.tanh
        elif after=='sigmoid':
            self.after = ops.sigmoid
        if before=='ReLU':
            self.before = nn.ReLU()
        elif before=='LReLU':
            self.before = nn.LeakyReLU(alpha=0.2)

    def construct(self, x):

        if hasattr(self, 'before'):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, 'after'):
            x = self.after(x)

        return x


class CvTi(nn.Cell):
    def __init__(self, in_channels, out_channels, before=None, after=None, kernel_size=4, stride=2,
                 padding=1, dilation=1, groups=1, bias=False):
        super(CvTi, self).__init__(auto_prefix=True)
        self.conv = nn.Conv2dTranspose(in_channels, out_channels, kernel_size, stride,
                                       pad_mode='pad', padding=padding, has_bias=bias, weight_init=init.Normal(0.0, 0.02))

        if after=='BN':
            self.after = nn.BatchNorm2d(out_channels)
        elif after=='Tanh':
            self.after = ops.tanh
        elif after=='sigmoid':
            self.after = ops.sigmoid

        if before=='ReLU':
            self.before = nn.ReLU()
        elif before=='LReLU':
            self.before = nn.LeakyReLU(alpha=0.2)

    def construct(self, x):

        if hasattr(self, 'before'):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, 'after'):
            x = self.after(x)

        return x

class BENet(nn.Cell):
    def __init__(self, in_channels: int = 3, out_channels: int = 3) -> None:
        super(BENet, self).__init__(auto_prefix=True)

        self.Cv0 = Cvi(in_channels, 64)

        self.Cv1 = Cvi(64, 128, before='LReLU', after='BN')

        self.Cv2 = Cvi(128, 256, before='LReLU', after='BN')

        self.Cv3 = Cvi(256, 512, before='LReLU', after='BN')

        self.Cv4 = Cvi(512, 512, before='LReLU', after='BN')

        self.Cv5 = Cvi(512, 512, before='LReLU')

        self.CvT6 = CvTi(512, 512, before='ReLU', after='BN')

        self.CvT7 = CvTi(1024, 512, before='ReLU', after='BN')

        self.CvT8 = CvTi(1024, 256, before='ReLU', after='BN')

        self.CvT9 = CvTi(512, 128, before='ReLU', after='BN')

        self.CvT10 = CvTi(256, 64, before='ReLU', after='BN')

        self.CvT11 = CvTi(128, out_channels, before='ReLU', after='Tanh')

    def construct(self, input):
        #encoder
        x0 = self.Cv0(input)
        x1 = self.Cv1(x0)
        x2 = self.Cv2(x1)
        x3 = self.Cv3(x2)
        x4_1 = self.Cv4(x3)
        x4_2 = self.Cv4(x4_1)
        x4_3 = self.Cv4(x4_2)
        x5 = self.Cv5(x4_3)

        #decoder
        x6 = self.CvT6(x5)

        cat1_1 = ops.cat([x6, x4_3], axis=1)
        x7_1 = self.CvT7(cat1_1)
        cat1_2 = ops.cat([x7_1, x4_2], axis=1)
        x7_2 = self.CvT7(cat1_2)
        cat1_3 = ops.cat([x7_2, x4_1], axis=1)
        x7_3 = self.CvT7(cat1_3)

        cat2 = ops.cat([x7_3, x3], axis=1)
        x8 = self.CvT8(cat2)

        cat3 = ops.cat([x8, x2], axis=1)
        x9 = self.CvT9(cat3)

        cat4 = ops.cat([x9, x1], axis=1)
        x10 = self.CvT10(cat4)

        cat5 = ops.cat([x10, x0], axis=1)
        out = self.CvT11(cat5)

        return out

class Discriminator(nn.Cell):
    def __init__(self, in_channels=6):
        super(Discriminator, self).__init__(auto_prefix=True)

        self.Cv0 = Cvi(in_channels, 64)

        self.Cv1 = Cvi(64, 128, before='LReLU', after='BN')

        self.Cv2 = Cvi(128, 256, before='LReLU', after='BN')

        self.Cv3 = Cvi(256, 512, before='LReLU', after='BN')

        self.Cv4 = Cvi(512, 1, before='LReLU', after='sigmoid')

    def construct(self, input):
        x0 = self.Cv0(input)
        x1 = self.Cv1(x0)
        x2 = self.Cv2(x1)
        x3 = self.Cv3(x2)
        out = self.Cv4(x3)

        return out


def discriminator(pretrained: bool=False, **kwargs: Any) -> Discriminator:
    model = Discriminator(**kwargs)
    if pretrained:
        state_dict = mindspore.load('./pretrained/pretrained_discriminator_for_srnet.ckpt')
        model.load_state_dict(fix_model_state_dict(state_dict))
    return model
