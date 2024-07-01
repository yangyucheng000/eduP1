from logging import getLogger
from typing import Optional

import mindspore.nn as nn
from ..dataset_csv import DATASET_CSVS

__all__ = ["get_criterion"]
logger = getLogger(__name__)


def get_criterion(
    loss_function_name: Optional[str] = None,
    device: Optional[str] = None,
) -> nn.Cell:

    if loss_function_name == 'L1':
        criterion = nn.L1Loss()         # .to(device)
    elif loss_function_name == 'GAN':
        criterion = [nn.L1Loss(), nn.BCEWithLogitsLoss()]
    else:
        criterion = nn.L1Loss()

    return criterion