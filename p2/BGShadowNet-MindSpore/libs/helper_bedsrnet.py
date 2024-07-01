import time
from logging import getLogger
from typing import Any, Dict, Optional, Tuple

import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.train import Model
from mindspore import value_and_grad
from libs.trainer import Trainer
from .VGG_loss import VGGNet
from .meter import AverageMeter, ProgressMeter
from .metric import calc_accuracy

__all__ = ["train", "evaluate"]

logger = getLogger(__name__)

def set_requires_grad(nets, requires_grad=False):
    for net in nets:
        if net is not None:
            for param in net.trainable_params():
                param.requires_grad = requires_grad
vgg = VGGNet()
def perceptual_loss(x, y):
    c = nn.MSELoss()
    fx1, fx2 = vgg(x)
    fy1, fy2 = vgg(y)
    m1 = c(fx1, fy1)
    m2 = c(fx2, fy2)
    loss = (m1+m2)*0.06
    return loss


def do_one_iteration(
    sample, generator: nn.Cell, refine_net: nn.Cell, discriminator: nn.Cell, benet: nn.Cell, criterion: Any,
    iter_type: str, lambda_dict: Dict, optimizerG: Optional[nn.Optimizer] = None,
    optimizerD: Optional[nn.Optimizer] = None, optimizerGen: Optional[nn.Optimizer] = None,
) -> Tuple[int, float, float, np.ndarray, np.ndarray]:

    if iter_type not in ["train", "evaluate"]:
        message = "iter_type must be either 'train' or 'evaluate'."
        logger.error(message)
        raise ValueError(message)

    if iter_type == "train" and (optimizerG is None or optimizerD is None):
        message = "optimizer must be set during training."
        logger.error(message)
        raise ValueError(message)

    Tensor = mindspore.Tensor

    x = sample[0]
    gt = sample[1]
    background, featureMap = benet(x)

    batch_size, c, h, w = x.shape

    discriminator.set_train(False)

    def forward_gen(gt, feature):
        confuse_result, confuseFeatureMap = generator(x, feature)
        G_L_confuse = criterion[0](gt, confuse_result)*0.2
        return G_L_confuse, confuse_result, confuseFeatureMap
    grad_fn_gen = mindspore.value_and_grad(forward_gen, None, optimizerGen.parameters, has_aux=True)
    def train_gen_step(gt, feature):
        (G_L_confuse, confuse_result, confuseFeatureMap), grad_gen = grad_fn_gen(gt, feature)
        optimizerGen(grad_gen)
        return G_L_confuse, confuse_result, list(confuseFeatureMap[0])
    G_L_confuse, confuse_result, confuseFeatureMap = train_gen_step(gt, featureMap)

    def forward_refine(gt, confuse_result, confuseFeatureMap):
        shadow_removal_image, _, _ = refine_net(confuse_result, background, x, confuseFeatureMap)

        fake = ops.cat([x, shadow_removal_image], axis=1)
        out_D_fake = discriminator(fake)
        label_D_real = Tensor(np.ones(out_D_fake.shape), dtype=mindspore.float32)

        G_L_GAN = criterion[1](out_D_fake, label_D_real)
        G_L_data = criterion[0](gt, shadow_removal_image)
        # G_L_confuse = criterion[0](gt, confuse_result)
        G_L_VGG = perceptual_loss(gt, shadow_removal_image)
        G_loss = lambda_dict["lambda1"] * G_L_data + lambda_dict["lambda2"] * G_L_GAN + G_L_VGG  # + 0.2 * G_L_confuse
        return G_loss, shadow_removal_image, confuse_result
    grad_fn_refine = mindspore.value_and_grad(forward_refine, None, optimizerG.parameters, has_aux=True)
    def train_refine_step(gt, confuse_result, confuseFeatureMap):
        (G_loss, shadow_removal_image, confuse_result), grads_refine = grad_fn_refine(gt, confuse_result, confuseFeatureMap)
        optimizerG(grads_refine)
        return G_loss, shadow_removal_image, confuse_result
    G_loss, shadow_removal_image, confuse_result = train_refine_step(gt, confuse_result, confuseFeatureMap)
    G_loss = G_loss + G_L_confuse

    discriminator.set_train(True)
    real = ops.cat([x, gt], axis=1)
    fake = ops.cat([x, shadow_removal_image], axis=1)

    def forward_fn_dis(data, train_fake=True):
        logits = discriminator(data)
        if train_fake:
            label = Tensor(np.zeros(logits.shape), dtype=mindspore.float32)
        else:
            label = Tensor(np.ones(logits.shape), dtype=mindspore.float32)
        label = mindspore.Parameter(label, requires_grad=True)
        loss = criterion[1](logits, label) * 0.1
        return loss, logits, label
    grad_dis = mindspore.value_and_grad(forward_fn_dis, None, weights=optimizerD.parameters, has_aux=True)
    def train_dis_step(data, label):
        (loss, logits, label), grads = grad_dis(data, label)
        optimizerD(grads)
        return loss, logits, label
    loss_D_real, _, label_D_real = train_dis_step(real, False)
    loss_D_fake, _, label_D_fake = train_dis_step(fake, True)

    D_L_GAN = loss_D_fake + loss_D_real
    # D_loss = lambda_dict["lambda2"] * D_L_GAN
    D_loss = D_L_GAN

    gt = gt.numpy()     # to("cpu")
    pred = shadow_removal_image.asnumpy()
    background = background.asnumpy()
    confuse_result = confuse_result.asnumpy()

    return batch_size, G_loss.numpy().item(), D_loss.numpy().item(), gt, pred, confuse_result


def train(
    loader,
    generator: nn.Cell,
    refine_net: nn.Cell,
    discriminator: nn.Cell,
    benet: nn.Cell,
    criterion: Any,
    lambda_dict: Dict,
    optimizerG: nn.Optimizer,
    optimizerD: nn.Optimizer,
    optimizerGen: nn.Optimizer,
    epoch: int,
    interval_of_progress: int = 50,
) -> Tuple[float, float, float]:

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    g_losses = AverageMeter("Loss", ":.4e")
    d_losses = AverageMeter("Loss", ":.4e")

    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, g_losses, d_losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []

    # switch to train mode
    generator.set_train(True)
    discriminator.set_train(True)
    refine_net.set_train(True)
    benet.set_train(False)

    end = time.time()
    for i, sample in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        batch_size, g_loss, d_loss, gt, pred, _ = do_one_iteration(
            sample, generator, refine_net, discriminator, benet, criterion, "train", lambda_dict,
            optimizerG, optimizerD, optimizerGen
        )

        g_losses.update(g_loss, batch_size)
        d_losses.update(d_loss, batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # show progress bar per 50 iteration
        if i != 0 and i % interval_of_progress == 0:
            progress.display(i)

    return g_losses.get_average(), d_losses.get_average()


def evaluate(
    loader, generator: nn.Cell, discriminator: nn.Cell, benet:nn.Cell,criterion: Any, lambda_dict: Dict, # device: str
) -> Tuple[float, float]:
    g_losses = AverageMeter("Loss", ":.4e")
    d_losses = AverageMeter("Loss", ":.4e")

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []

    # switch to evaluate mode
    generator.eval()
    discriminator.eval()

    # with torch.no_grad():
    for sample in loader:
        batch_size, g_loss, d_loss, gt, pred = do_one_iteration(        # device,
            sample, generator, discriminator, benet, criterion, "evaluate", lambda_dict
        )

        g_losses.update(g_loss, batch_size)
        d_losses.update(d_loss, batch_size)

    return g_losses.get_average(), d_losses.get_average()


def test_net(
    sample, generator: nn.Cell, refine_net: nn.Cell, discriminator: nn.Cell, benet: nn.Cell, criterion: Any,
    iter_type: str, lambda_dict: Dict,
):
    Tensor = mindspore.Tensor
    x = sample["img"]
    gt = sample["gt"]
    background, featureMap = benet(x)

    batch_size, c, h, w = x.shape

    # compute output and loss
    confuse_result, confuseFeatureMap = generator(x, featureMap)
    confuse_result = confuse_result
    shadow_removal_image, _, _ = refine_net(confuse_result, background, x, confuseFeatureMap)

    fake = ops.cat([x, shadow_removal_image], axis=1)
    real = ops.cat([x, gt], axis=1)

    out_D_fake = discriminator(fake)
    out_D_real = discriminator(real)

    label_D_fake = mindspore.Tensor(np.zeros(out_D_fake.shape), dtype=mindspore.float32)
    label_D_real = Tensor(np.ones(out_D_fake.shape), dtype=mindspore.float32)
    loss_D_fake = criterion[1](out_D_fake, label_D_fake)
    loss_D_real = criterion[1](out_D_real, label_D_real)
    D_L_GAN = loss_D_fake + loss_D_real
    D_loss = lambda_dict["lambda2"] * D_L_GAN

    fake = ops.cat([x, shadow_removal_image], axis=1)
    out_D_fake = discriminator(fake)

    G_L_GAN = criterion[1](out_D_fake, label_D_real)
    G_L_data = criterion[0](gt, shadow_removal_image)
    G_L_confuse = criterion[0](gt, confuse_result)
    G_L_VGG = perceptual_loss(gt, shadow_removal_image)

    G_loss = lambda_dict["lambda1"] * G_L_data + lambda_dict[
        "lambda2"] * G_L_GAN + G_L_VGG + 0.2 * G_L_confuse  # 粗网络的loss真的有意义吗?

    gt = gt.numpy()  # to("cpu")
    pred = shadow_removal_image.asnumpy()
    background = background.asnumpy()
    confuse_result = confuse_result.asnumpy()

    return batch_size, G_loss.value(), D_loss.value(), gt, pred, confuse_result
