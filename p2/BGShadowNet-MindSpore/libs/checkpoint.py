import os
from logging import getLogger
from typing import Tuple

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

logger = getLogger(__name__)


def save_checkpoint(
    result_path: str,
    epoch: int,
    model: nn.Cell,
    optimizer: nn.Optimizer,
    best_loss: float,
) -> None:

    save_states = {
        "epoch": epoch,
        "state_dict": model,             #.state_dict(),
        "optimizer": optimizer,           # .state_dict(),
        "best_loss": best_loss,
    }

    mindspore.save_checkpoint(model, os.path.join(result_path, "checkpoint.ckpt"))
    logger.debug("successfully saved the ckeckpoint.")

def save_checkpoint_BEDSRNet(
    result_path: str,
    epoch: int,
    generator: nn.Cell,
    discriminator: nn.Cell,
    optimizerG: nn.Optimizer,
    optimizerD: nn.Optimizer,
    optimizerGen: nn.Optimizer,
    best_g_loss: float,
    best_d_loss: float,
) -> None:

    save_states = {
        "epoch": epoch,
        "state_dictG": generator,
        "optimizerG": optimizerGen,
        "best_g_loss": best_g_loss,
    }

    mindspore.save_checkpoint(generator, os.path.join(result_path, "pretrained_g.ckpt"))
    logger.debug("successfully saved the generator's checkpoint.")

    save_states = {
        "epoch": epoch,
        "state_dictD": discriminator,
        "optimizerD": optimizerD,
        "best_d_loss": best_d_loss,
    }

    mindspore.save_checkpoint(discriminator, os.path.join(result_path, "pretrained_d.ckpt"))
    logger.debug("successfully saved the discriminator's checkpoint.")


def resume(
    resume_path: str, model: nn.Cell, optimizer: nn.Optimizer
) -> Tuple[int, nn.Cell, nn.Optimizer, float]:
    try:
        checkpoint = mindspore.load_checkpoint(resume_path)
        logger.info("loading checkpoint {}".format(resume_path))
    except FileNotFoundError("there is no checkpoint at the result folder.") as e:
        logger.exception(f"{e}")

    begin_epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]
    model.load_state_dict(checkpoint["state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer"])

    logger.info("training will start from {} epoch".format(begin_epoch))

    return begin_epoch, model, optimizer, best_loss

def resume_BEDSRNet(
    resume_path: str, generator: nn.Cell, discriminator: nn.Cell, optimizerG: nn.Optimizer, optimizerD: nn.Optimizer,
    optimizerGen: nn.Optimizer
) -> Tuple[int, nn.Cell, nn.Cell, nn.Optimizer, nn.Optimizer, float, float]:
    try:
        checkpoint_g = mindspore.load(os.path.join(resume_path + 'g_checkpoint.ckpt'))
        logger.info("loading checkpoint {}".format(os.path.join(resume_path + 'g_checkpoint.ckpt')))
        checkpoint_d = mindspore.load(os.path.join(resume_path + 'd_checkpoint.ckpt'))
        logger.info("loading checkpoint {}".format(os.path.join(resume_path + 'g_checkpoint.ckpt')))
    except FileNotFoundError("there is no checkpoint at the result folder.") as e:
        logger.exception(f"{e}")

    begin_epoch = checkpoint_g["epoch"]
    best_g_loss = checkpoint_g["best_g_loss"]
    best_d_loss = checkpoint_d["best_d_loss"]

    mindspore.load_param_into_net(generator, checkpoint_g["state_dict"])
    mindspore.load_param_into_net(discriminator, checkpoint_d["state_dict"])

    mindspore.load_param_into_net(optimizerG, checkpoint_g["optimizer"])
    mindspore.load_param_into_net(optimizerD, checkpoint_d["optimizer"])
    mindspore.load_param_into_net(optimizerGen, checkpoint_d["optimizer"])

    logger.info("training will start from {} epoch".format(begin_epoch))

    return begin_epoch, generator, discriminator, optimizerG, optimizerD, optimizerGen, best_g_loss, best_d_loss