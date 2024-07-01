from logging import getLogger
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import mindspore as ms

logger = getLogger(__name__)

ms.set_context(device_target='GPU', device_id=0)
