import mindspore.nn as nn
import mindspore_hub as mshub


class VGGNet(nn.Cell):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__(auto_prefix=True)
        self.select = [3, 8, 13, 22, 31]
        model = 'mindspore/1.9/vgg19_cifar10'
        self.vgg = mshub.load(model, num_classes=10)
        self.vgg.set_train(False)

    def construct(self, x):
        """Extract multiple convolutional feature maps."""
        features = []
        i = 0
        for layer in self.vgg.layers:
            x = layer(x)
            if i in self.select:
                features.append(x)
            i += 1
        return features[0], features[1], features[2], features[3], features[4]