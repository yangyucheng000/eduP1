import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
from logging import getLogger
from typing import Any, Dict, Optional, Tuple
from .models.models import weights_init
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import mindspore
import mindspore.nn as nn

from .meter import AverageMeter, ProgressMeter
from .metric import calc_accuracy
from .VGG_loss_5 import VGGNet


__all__ = ["train", "evaluate"]

logger = getLogger(__name__)
vgg = VGGNet().set_train(False)
def perceptual_loss(x, y):
    c = nn.L1Loss()
    fx1, fx2, fx3, fx4, fx5 = vgg(x)
    fy1, fy2, fy3, fy4, fy5 = vgg(y)
    m1 = c(fx1, fy1)/2.6
    m2 = c(fx2, fy2)/4.8
    m3 = c(fx3, fy3)/3.7
    m4 = c(fx4, fy4)/5.6
    m5 = c(fx5, fy5)*10/1.5
    loss = m1+m2+m3+m4+m5
    return loss

def color_loss(x,y):
    c = nn.MSELoss()
    conv1 = nn.Conv2d(3, 3, (3, 3)).cuda()
    conv1.apply(weights_init('gaussian'))
    x1 = conv1(x)
    y1 = conv1(y)
    loss = c(x1,y1)*10
    return loss

def do_one_iteration(
    sample: Dict[str, Any],
    model: nn.Cell,
    criterion: Any,
    iter_type: str,
    grad_fn,
    optimizer: Optional[nn.Optimizer] = None,
) -> Tuple[int, float, np.ndarray, np.ndarray]:

    if iter_type not in ["train", "evaluate"]:
        message = "iter_type must be either 'train' or 'evaluate'."
        logger.error(message)
        raise ValueError(message)

    if iter_type == "train" and optimizer is None:
        message = "optimizer must be set during training."
        logger.error(message)
        raise ValueError(message)

    x = sample["img"]       # .to(device)
    t = sample["back_img"]

    batch_size = x.shape[0]

    # compute output and loss
    if iter_type == "test":
        output, _ = model(x)
        loss = perceptual_loss(output, t)+criterion(output, t)
    if iter_type == "train":
        (loss, _), grads = grad_fn(x, t)
        optimizer(grads)

    gt = t.numpy()
    pred = output.numpy()

    return batch_size, loss.item(), gt, pred


def train(
    loader,
    model: nn.Cell,
    criterion: Any,
    optimizer: nn.Optimizer,
    epoch: int,
    interval_of_progress: int = 50,
) -> Tuple[float, float, float]:

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")

    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses],        # top1
        prefix="Epoch: [{}]".format(epoch),
    )

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []

    # switch to train mode
    model.set_train(True)

    def forward_fn(x, t):
        output, _ = model(x)
        loss = perceptual_loss(output, t)+criterion(output, t)
        return loss, output
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(x, t):
        (loss, output), grads = grad_fn(x, t)
        optimizer(grads)
        return loss, output

    end = time.time()
    i = 0
    loader_iter = loader.create_dict_iterator()
    for sample in loader_iter:
        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = loader.get_batch_size()
        loss, pred = train_step(sample["img"], sample["back_img"])
        loss = loss.numpy()
        loss = loss.item()
        gt = sample["back_img"].numpy()
        pred = pred.numpy()

        losses.update(loss, batch_size)

        # save the ground truths and predictions in lists
        gts += list(gt)
        preds += list(pred)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # show progress bar per 50 iteration
        if i != 0 and i % interval_of_progress == 0:
            progress.display(i)
            i += 1

    return losses.get_average()


def evaluate(
    loader, model: nn.Cell, criterion: Any,
) -> Tuple[float]:
    losses = AverageMeter("Loss", ":.4e")

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []

    # switch to evaluate mode
    model.eval()

    # with torch.no_grad():
    for sample in loader:
        batch_size, loss, gt, pred = do_one_iteration(
            sample, model, criterion, "evaluate"
        )

        losses.update(loss, batch_size)

        # keep predicted results and gts for calculate F1 Score
        gts += list(gt)
        preds += list(pred)

    return losses.get_average()
