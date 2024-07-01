import cv2
import os

import mindspore
from typing import Any, Dict, Optional, Tuple

from libs.models.models import Discriminator

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from libs.fix_weight_dict import fix_model_state_dict
from libs.models.tiramisu import *
from libs.models.tiramisuskip import *
from albumentations import (
    Compose,
    Normalize,
    Resize,
    Transpose
)
from libs.models.STLtiramisuskip import *
from utils.visualize import visualize, reverse_normalize
from libs.dataset import get_dataloader
from libs.loss_fn import get_criterion
from libs.helper_bedsrnet import test_net

if __name__ == '__main__':

    def convert_show_image(tensor, idx=None):
        if tensor.shape[1] == 3:
            img = reverse_normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif tensor.shape[1] == 1:
            img = tensor * 0.5 + 0.5

        if idx is not None:
            img = (img[idx].transpose(1, 2, 0) * 255).astype(np.uint8)
        else:
            img = (img.squeeze(axis=0).transpose(1, 2, 0) * 255).astype(np.uint8)

        return img

    test_transform = Compose([Resize(256, 256), Normalize(mean=(0.5,), std=(0.5,))])
    test_loader = get_dataloader(
        "Jung",
        "bedsrnet",
        "test",
        batch_size=1,
        shuffle=False,
        num_workers=2,
        transform=test_transform,
    )
    mindspore.set_context(device_target='GPU', device_id=0)
    benet = FCDenseNet57(3)
    generator = FCDenseSkipNet57(3)
    discriminator = Discriminator(6)

    benet_weights = mindspore.load_checkpoint('pretrained/benet.ckpt')
    fix_model_state_dict(benet_weights)
    mindspore.load_param_into_net(benet, benet_weights)

    generator_weights = mindspore.load_checkpoint('pretrained/generator.ckpt')
    fix_model_state_dict(generator_weights)
    mindspore.load_param_into_net(generator, generator_weights)

    discriminator_weights = mindspore.load_checkpoint('pretrained/discriminator.ckpt')
    fix_model_state_dict(discriminator_weights)
    mindspore.load_param_into_net(discriminator, discriminator_weights)

    refine_net = STLFCDenseSkipNet57(6)
    refine_weights = mindspore.load_checkpoint('pretrained/refine_net.ckpt')
    fix_model_state_dict(refine_weights)
    mindspore.load_param_into_net(refine_net, refine_weights)
    generator.set_train(False)
    discriminator.set_train(False)
    benet.set_train(False)
    refine_net.set_train(False)
    criterion = get_criterion("GAN")
    lambda_dict = {"lambda1": 1.0, "lambda2": 0.01}


    def check_dir():
        if not os.path.exists('./test_result'):
            os.mkdir('./test_result')
        if not os.path.exists('./test_result/test'):
            os.mkdir('./test_result/test')
    check_dir()

    for sample in test_loader.create_dict_iterator():
        _, _, _, gt, pred, coares_result = test_net(sample, generator, refine_net, discriminator, benet,
                                                            criterion, "evaluate", lambda_dict)
        img_name = str(sample['img_path']).split('/')[-1].split('.')[0] + '.png'
        shadow_removal = convert_show_image(np.array(pred))
        cv2.imwrite('./test_result/test/' + img_name, shadow_removal)
