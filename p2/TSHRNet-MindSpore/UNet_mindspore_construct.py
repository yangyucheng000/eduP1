from mindspore import nn
from mindspore.common.initializer import Normal
from mindspore.ops import cat


class Cvi(nn.Cell):
    def __init__(self, in_channels, out_channels, before=None, after=None, kernel_size=4, stride=2,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Cvi, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, pad_mode='pad', padding=padding,
                              dilation=dilation, group=groups, has_bias=bias,
                              weight_init=Normal(0.02))
        if after == 'BN':
            self.after = nn.BatchNorm2d(out_channels)
        elif after == 'Tanh':
            self.after = nn.Tanh()
        elif after == 'sigmoid':
            self.after = nn.Sigmoid()

        if before == 'ReLU':
            self.before = nn.ReLU()
        elif before == 'LReLU':
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
                 padding=1, groups=1, bias=False):
        super(CvTi, self).__init__()
        self.bias = bias
        self.groups = groups
        self.conv = nn.Conv2dTranspose(in_channels, out_channels, kernel_size, stride=stride, pad_mode='pad',
                                       padding=padding, weight_init=Normal(0.02), has_bias=True)

        if after == 'BN':
            self.after = nn.BatchNorm2d(out_channels)
        elif after == 'Tanh':
            self.after = nn.Tanh()
        elif after == 'sigmoid':
            self.after = nn.Sigmoid()

        if before == 'ReLU':
            self.before = nn.ReLU()
        elif before == 'LReLU':
            self.before = nn.LeakyReLU(alpha=0.2)

    def construct(self, x):

        if hasattr(self, 'before'):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, 'after'):
            x = self.after(x)

        return x


class UNet(nn.Cell):
    def __init__(self, input_channels=3, output_channels=1):
        super(UNet, self).__init__()
        self.loss = nn.MSELoss()
        self.Cv0 = Cvi(input_channels, 64)
        self.Cv1 = Cvi(64, 128, before='LReLU', after='BN', dilation=1)
        self.Cv2 = Cvi(128, 256, before='LReLU', after='BN', dilation=1)
        self.Cv3 = Cvi(256, 512, before='LReLU', after='BN', dilation=1)
        self.Cv4 = Cvi(512, 512, before='LReLU', after='BN', dilation=1)
        self.Cv5 = Cvi(512, 512, before='LReLU', dilation=1)

        self.CvT6 = CvTi(512, 512, before='ReLU', after='BN')
        self.CvT7 = CvTi(1024, 512, before='ReLU', after='BN')
        self.CvT8 = CvTi(1024, 256, before='ReLU', after='BN')
        self.CvT9 = CvTi(512, 128, before='ReLU', after='BN')
        self.CvT10 = CvTi(256, 64, before='ReLU', after='BN')
        self.CvT11 = CvTi(128, output_channels, before='ReLU', after='Tanh')

    def construct(self, x):
        # encoder
        x0 = self.Cv0(x)
        x1 = self.Cv1(x0)
        x2 = self.Cv2(x1)
        x3 = self.Cv3(x2)
        x4_1 = self.Cv4(x3)
        x4_2 = self.Cv4(x4_1)
        x4_3 = self.Cv4(x4_2)
        x5 = self.Cv5(x4_3)

        # decoder
        x6 = self.CvT6(x5)

        cat1_1 = cat([x6, x4_3], axis=1)
        x7_1 = self.CvT7(cat1_1)
        cat1_2 = cat([x7_1, x4_2], axis=1)
        x7_2 = self.CvT7(cat1_2)
        cat1_3 = cat([x7_2, x4_1], axis=1)
        x7_3 = self.CvT7(cat1_3)

        cat2 = cat([x7_3, x3], axis=1)
        x8 = self.CvT8(cat2)

        cat3 = cat([x8, x2], axis=1)
        x9 = self.CvT9(cat3)

        cat4 = cat([x9, x1], axis=1)
        x10 = self.CvT10(cat4)

        cat5 = cat([x10, x0], axis=1)
        out = self.CvT11(cat5)

        return out
