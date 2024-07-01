from utils.SpectralNorm import SpectralNorm as SpectralNorm
import math
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.initializer as init
from mindspore.common.initializer import initializer

if __name__ == '__main__':
    from layerskip import *
    from STLNet import *
else:
    from .layerskip import *
    from .STLNet import *


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.Normal(m.weight, 0.0, 0.02)
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
        super(Cvi, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='pad',
                              padding=padding, dilation=dilation, group=groups, has_bias=bias)
        if after == 'BN':
            self.after = nn.BatchNorm2d(out_channels, momentum=0.1)
        elif after == 'Tanh':
            self.after = ops.tanh
        elif after == 'sigmoid':
            self.after = ops.sigmoid

        if before == 'ReLU':
            self.before = nn.ReLU()  # inplace=True
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
                 padding=1, dilation=1, groups=1, bias=False):
        super(CvTi, self).__init__()
        self.conv = nn.Conv2dTranspose(in_channels, out_channels, kernel_size, stride,
                                       pad_mode='pad', padding=padding, has_bias=bias)
        if after == 'BN':
            self.after = nn.BatchNorm2d(out_channels, momentum=0.1)
        elif after == 'Tanh':
            self.after = ops.tanh
        elif after == 'sigmoid':
            self.after = ops.sigmoid

        if before == 'ReLU':
            self.before = nn.ReLU()  # inplace=True
        elif before == 'LReLU':
            self.before = nn.LeakyReLU(alpha=0.2)

    def construct(self, x):

        if hasattr(self, 'before'):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, 'after'):
            x = self.after(x)

        return x


def get_norm(name, ch):
    if name == 'bn':
        return nn.BatchNorm2d(ch, momentum=0.1)
    elif name == 'in':
        return nn.InstanceNorm2d(ch, affine=False)
    else:
        raise NotImplementedError('Normalization %s not implemented' % name)


class Tanh2(nn.Cell):
    def __init__(self):
        super(Tanh2, self).__init__()
        self.tanh = nn.Tanh()

    def construct(self, x):
        return (self.tanh(x) + 1) / 2


def get_activ(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'lrelu':
        return nn.LeakyReLU(0.2)
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'tanh2':
        return Tanh2()
    else:
        raise NotImplementedError('Activation %s not implemented' % name)


class ResidualBlock(nn.Cell):
    def __init__(self, inc, outc=None, kernel=3, stride=1, activ='lrelu', norm='bn', sn=False):
        super(ResidualBlock, self).__init__()

        if outc is None:
            outc = inc // stride

        self.activ = get_activ(activ)
        pad = kernel // 2
        if sn:
            self.input = SpectralNorm(nn.Conv2d(inc, outc, 1, 1, padding=0, pad_mode='pad', has_bias=True))
            self.blocks = nn.SequentialCell(SpectralNorm(nn.Conv2d(inc, outc, kernel, 1, pad_mode='pad', padding=pad, has_bias=True)),
                                            get_norm(norm, outc),
                                            nn.LeakyReLU(0.2),
                                            SpectralNorm(nn.Conv2d(outc, outc, kernel, 1, pad_mode='pad', padding=pad, has_bias=True)),
                                            get_norm(norm, outc))
        else:
            self.input = nn.Conv2d(inc, outc, 1, 1, padding=0, pad_mode='pad', has_bias=True)
            self.blocks = nn.SequentialCell(nn.Conv2d(inc, outc, kernel, 1, pad_mode='pad', padding=1, has_bias=True),  # kernel
                                            get_norm(norm, outc),
                                            nn.LeakyReLU(0.2),
                                            nn.Conv2d(outc, outc, kernel, 1, pad_mode='pad', padding=1, has_bias=True),
                                            get_norm(norm, outc))

    def construct(self, x):
        return self.activ(self.blocks(x) + self.input(x))


def ConvBlock(inc, outc, ks=3, s=1, p=0, activ='lrelu', norm='bn', res=0, resk=3, bn=True, sn=False):
    conv = nn.Conv2d(inc, outc, ks, s, pad_mode='pad', padding=p, has_bias=True)
    if sn:
        conv = SpectralNorm(conv)
    blocks = [conv]
    if bn:
        blocks.append(get_norm(norm, outc))
    if activ is not None:
        blocks.append(get_activ(activ))
    for i in range(res):
        blocks.append(ResidualBlock(outc, activ=activ, kernel=resk, norm=norm, sn=sn))
    return nn.SequentialCell(*blocks)


class FCDenseNet(nn.Cell):
    def __init__(self, in_channels=6, down_blocks=(5, 5, 5, 5, 5),
                 up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, n_classes=12):
        super().__init__()
        self.down_blocks = down_blocks
        self.STL = STL((out_chans_first_conv + growth_rate * (down_blocks[0] + down_blocks[1])) * 3)
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        ## First Convolution ##
        self.firstconv = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_chans_first_conv, kernel_size=3,
                                   stride=1, padding=1, pad_mode='pad', has_bias=True)
        self.STLfirstConv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_chans_first_conv + growth_rate * (
                                                  down_blocks[0] + down_blocks[1]),
                                      kernel_size=3,
                                      stride=2, padding=1, pad_mode='pad', has_bias=True)
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################
        self.denseBlocksDown = nn.CellList()
        self.transDownBlocks = nn.CellList()
        for i in range(len(down_blocks)):
            self.denseBlocksDown.insert_child_to_cell(str(i),
                                                      DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate * down_blocks[i])
            if (i == 1):
                skip_connection_channel_counts.insert(0, cur_channels_count + 96)  # 256为STL输出通道数512的一半
            else:
                skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.insert_child_to_cell(str(i), TransitionDown(cur_channels_count))
        self.denseBlocksDown = self.denseBlocksDown
        self.transDownBlocks = self.transDownBlocks

        #####################
        #     Bottleneck    #
        #####################
        self.bottleneck = Bottleneck(cur_channels_count, growth_rate, bottleneck_layers)
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################
        self.transUpBlocks = nn.CellList()
        self.denseBlocksUp = nn.CellList()
        for i in range(len(up_blocks)):
            if i != len(up_blocks) - 1:
                self.transUpBlocks.insert_child_to_cell(str(i),
                                                        TransitionUp(prev_block_channels, prev_block_channels))
                cur_channels_count = prev_block_channels + skip_connection_channel_counts[i] * 3
                if i == 3:
                    cur_channels_count = 672
                self.denseBlocksUp.insert_child_to_cell(str(i), DenseBlock(
                                                        cur_channels_count, growth_rate, up_blocks[i], upsample=True))
                prev_block_channels = growth_rate * up_blocks[i]
                cur_channels_count += prev_block_channels
            else:
                self.transUpBlocks.insert_child_to_cell(str(i), TransitionUp(
                                                        prev_block_channels, prev_block_channels))
                cur_channels_count = 336  # prev_block_channels + skip_connection_channel_counts[-1]*2
                self.denseBlocksUp.insert_child_to_cell(str(i), DenseBlock(
                                                        cur_channels_count, growth_rate, up_blocks[-1], upsample=False))
                cur_channels_count += growth_rate * up_blocks[-1]
        self.transUpBlocks = self.transUpBlocks
        self.denseBlocksUp = self.denseBlocksUp

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
                                   out_channels=3, kernel_size=1, stride=1,
                                   padding=0, has_bias=True)
        self.Cv0 = Cvi(3, 96, kernel_size=3, stride=1, padding=1)

        self.Cv1 = Cvi(96, 144, before='LReLU', after='BN')

        self.Cv2 = Cvi(144, 192, before='LReLU', after='BN')

        self.Cv3 = Cvi(192, 240, before='LReLU', after='BN')

        self.Cv4 = Cvi(240, 288, before='LReLU', after='BN')

        self.att0 = nn.SequentialCell(ConvBlock(96 * 2, 96 * 2, 3, 1, 1, sn=True),
                                      ResidualBlock(96 * 2, 96 * 2, activ='sigmoid', sn=True))
        self.att1 = nn.SequentialCell(ConvBlock(144 * 2, 144 * 2, 3, 1, 1, sn=True),
                                      ResidualBlock(144 * 2, 144 * 2, activ='sigmoid', sn=True))
        self.att2 = nn.SequentialCell(ConvBlock(192 * 2, 192 * 2, 3, 1, 1, sn=True),
                                      ResidualBlock(192 * 2, 192 * 2, activ='sigmoid', sn=True))
        self.att3 = nn.SequentialCell(ConvBlock(240 * 2, 240 * 2, 3, 1, 1, sn=True),
                                      ResidualBlock(240 * 2, 240 * 2, activ='sigmoid', sn=True))
        self.att4 = nn.SequentialCell(ConvBlock(288 * 2, 288 * 2, 3, 1, 1, sn=True),
                                      ResidualBlock(288 * 2, 288 * 2, activ='sigmoid', sn=True))

    def construct(self, confuse_result, background, shadow_img, featureMaps):
        ms_res_dict = {}
        x = ops.cat([confuse_result, shadow_img], axis=1)
        background_feature = []
        back1 = self.Cv0(background)
        background_feature.append(back1)
        back2 = self.Cv1(back1)
        background_feature.append(back2)
        back3 = self.Cv2(back2)
        background_feature.append(back3)
        back4 = self.Cv3(back3)
        background_feature.append(back4)
        back5 = self.Cv4(back4)
        background_feature.append(back5)
        out = self.firstconv(x)
        STLFirst = self.STLfirstConv(x)

        ms_res_dict.update({"back1":back1, "back2":back2, "back3":back3,
                   "back4":back4, 'back5':back5, "firstconv":out, "STLFirst":STLFirst})

        skip_connections = []
        newFeatureMap = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)

            ms_res_dict["denseBlockDown" + str(i)] = out

            background_featuremap = background_feature[i]
            skip = ops.cat((out, background_featuremap), 1)
            att = getattr(self, "att{}".format(i))(skip)

            ms_res_dict["att" + str(i)] = att

            skip = skip * att
            skip_connections.append(skip)
            newFeatureMap.append(out)
            out = self.transDownBlocks[i](out)

            ms_res_dict["transDownBlocks" + str(i)] = out

        STLinput = ops.cat([STLFirst, skip_connections[1]], axis=1)
        STLresult, stl_res_dict = self.STL(STLinput)
        ms_res_dict["STLresult"] = STLresult
        ms_res_dict.update(stl_res_dict)
        skip_connections[1] = ops.cat([skip_connections[1], STLresult], axis=1)
        out = self.bottleneck(out)
        ms_res_dict["bottleneck"] = out
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            featureMap = featureMaps.pop()
            out = self.transUpBlocks[i](out, skip, featureMap)
            ms_res_dict["transUpBlocks" + str(i)] = out
            out = self.denseBlocksUp[i](out)
            ms_res_dict["denseBlocksUp" + str(i)] = out

        out = self.finalConv(out)
        ms_res_dict["finalConv"] = out
        return out, newFeatureMap, ms_res_dict


def STLFCDenseSkipNet57(in_channels=6):
    return FCDenseNet(
        in_channels=in_channels, down_blocks=(4, 4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=12,
        growth_rate=12, out_chans_first_conv=48, n_classes=3)


if __name__ == '__main__':
    size = (3, 3, 256, 256)
    input = ops.Ones(size)
    model = STLFCDenseSkipNet57(3)
    output = model(input)
