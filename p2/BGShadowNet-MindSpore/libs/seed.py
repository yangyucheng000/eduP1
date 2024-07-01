import random
from logging import getLogger

import mindspore
import numpy as np
import os

logger = getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    mindspore.set_seed(seed)

    logger.info("Finished setting up seed.")