import mindspore
import mindspore.nn as nn
if __name__ == '__main__':
    from layerskip import *
else:
    from .layerskip import *


class FCDenseNet(nn.Cell):
    def __init__(self, in_channels=6, down_blocks=(5,5,5,5,5),
                 up_blocks=(5,5,5,5,5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, n_classes=12):
        super().__init__(auto_prefix=True)
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        ## First Convolution ##
        self.firstconv = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_chans_first_conv, kernel_size=3,
                                   stride=1, padding=1, pad_mode='pad', has_bias=True)
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
            if i != len(up_blocks)-1:
                self.transUpBlocks.insert_child_to_cell(str(i), TransitionUp(prev_block_channels, prev_block_channels))
                cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]*2
                self.denseBlocksUp.insert_child_to_cell(str(i), DenseBlock(
                    cur_channels_count, growth_rate, up_blocks[i],
                    upsample=True))
                prev_block_channels = growth_rate * up_blocks[i]
                cur_channels_count += prev_block_channels
            else:
                self.transUpBlocks.insert_child_to_cell(str(i), TransitionUp(
                    prev_block_channels, prev_block_channels))
                cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]*2

                self.denseBlocksUp.insert_child_to_cell(str(i), DenseBlock(
                    cur_channels_count, growth_rate, up_blocks[-1],
                    upsample=False))
                cur_channels_count += growth_rate * up_blocks[-1]
        self.transUpBlocks = self.transUpBlocks
        self.denseBlocksUp = self.denseBlocksUp

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=3, kernel_size=1, stride=1,
                   padding=0, has_bias=True)

    def construct(self, x, featureMaps):
        out = self.firstconv(x)
        skip_connections = []
        newFeatureMap = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            newFeatureMap.append(out)
            out = self.transDownBlocks[i](out)
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            featureMap = featureMaps.pop()
            out = self.transUpBlocks[i](out, skip,featureMap)
            out = self.denseBlocksUp[i](out)
        out = self.finalConv(out)
        return out, newFeatureMap


def FCDenseSkipNet57(in_channels=3):
    return FCDenseNet(
        in_channels=in_channels, down_blocks=(4, 4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=16,
        growth_rate=12, out_chans_first_conv=48, n_classes=3)

if __name__ == '__main__':
    #Generator test
    size = (3, 3, 256, 256)
    input = mindspore.ops.ones(size)
    model = FCDenseNet67(3)
    output = model(input)
    print(input.shape)
