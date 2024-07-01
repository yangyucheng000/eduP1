import mindspore
import mindspore.nn as nn


class DenseLayer(nn.SequentialCell):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels, momentum=0.1)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3,
                                          stride=1, padding=1, pad_mode='pad', has_bias=True)
        self.cell_list = [self.norm, self.relu, self.conv]

    def construct(self, x):
        return super().construct(x)


class DenseBlock(nn.Cell):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super().__init__(auto_prefix=True)
        self.upsample = upsample
        self.layers = nn.CellList([DenseLayer(
            in_channels + i*growth_rate, growth_rate)
            for i in range(n_layers)])

    def construct(self, x):
        if self.upsample:
            new_features = []
            #we pass all previous activations into each dense layer normally
            #But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = mindspore.ops.cat([x, out], 1)
                new_features.append(out)
            return mindspore.ops.cat(new_features,1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = mindspore.ops.cat([x, out], 1)
            return x


class TransitionDown(nn.SequentialCell):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_features=in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, in_channels,
                                          kernel_size=1, stride=1,
                                          padding=0, has_bias=True)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.cell_list = [self.norm, self.relu, self.conv, self.maxpool]

    def construct(self, x):
        return super().construct(x)


class TransitionUp(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super().__init__(auto_prefix=True)
        self.convTrans = nn.Conv2dTranspose(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=2, padding=0, pad_mode='pad', has_bias=True)

    def construct(self, x, skip, featureMap):
        out = self.convTrans(x)
        out = center_crop(out, skip.shape[2], skip.shape[3])
        out = mindspore.ops.cat([out, skip, featureMap], 1)
        return out


class Bottleneck(nn.SequentialCell):
    def __init__(self, in_channels, growth_rate, n_layers):
        super().__init__()
        self.bottleneck = DenseBlock(in_channels, growth_rate, n_layers, upsample=True)
        self.cell_list = [self.bottleneck]

    def construct(self, x):
        return super().construct(x)


def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.shape
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]
