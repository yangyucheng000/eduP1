from logging import getLogger
from typing import Any, Dict, Optional
from PIL import Image
import pandas as pd
import cv2

import mindspore
from mindspore.dataset import GeneratorDataset      # no Dataloader

import albumentations as A
import numpy as np
from libs.dataset_csv import DATASET_CSVS

__all__ = ["get_dataloader"]

logger = getLogger(__name__)


def get_dataloader(
    dataset_name: str,
    train_model: str,
    split: str,         # train
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool = False,
    transform: Optional[A.Compose] = None,
):
    if dataset_name not in DATASET_CSVS:
        message = f"dataset_name should be selected from {list(DATASET_CSVS.keys())}."
        logger.error(message)
        raise ValueError(message)

    if train_model not in ["benet", "bedsrnet", "stcgan-be"]:
        message = f"dataset_name should be selected from ['benet', 'srnet', 'stcgan-be']."
        logger.error(message)
        raise ValueError(message)

    if split not in ["train", "val", "test"]:
        message = "split should be selected from ['train', 'val', 'test']."
        logger.error(message)
        raise ValueError(message)

    logger.info(f"Dataset: {dataset_name}\tSplit: {split}\tBatch size: {batch_size}.")

    csv_file = getattr(DATASET_CSVS[dataset_name], split)
    if train_model == "benet":
        data = BackGroundDataset(csv_file, transform=transform)
        # col = ["img", "img_path", "back_img"]
        col = ["img", "back_img"]
    elif train_model == "bedsrnet":
        data = ShadowDocumentDataset(csv_file, transform=transform)
        col = ["img", "gt", "img_path"]
    elif train_model == "stcgan-be":
        data = ShadowDocumentDataset(csv_file, transform=transform)
        col = ["img", "gt", "img_path"]

    dataloader = GeneratorDataset(
        source=data,
        column_names=col,
        shuffle=shuffle,
        num_parallel_workers=num_workers,
    )
    dataloader = dataloader.batch(batch_size=batch_size, drop_remainder=drop_last)
    return dataloader


class BackGroundDataset:
    def __init__(self, csv_file: str, transform: Optional[A.Compose] = None):
        try:
            self.df = pd.read_csv(csv_file)
        except FileNotFoundError("csv file not found.") as e:
            logger.exception(f"{e}")

        self.transform = transform

        logger.info(f"the number of samples: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["img"]
        background_path = self.df.iloc[idx]["back_gt"]
       
        img = cv2.imread(img_path)
        back_img = cv2.imread(background_path)
       
        images = np.concatenate([img, back_img], axis=2)

        if self.transform is not None:
            res = self.transform(image=images)['image']
            res = np.transpose(res, (2, 0, 1))
            img = res[:3,:,:]
            back_img = res[3:,:,:]
        return img, back_img


class ShadowDocumentDataset():
    def __init__(self, csv_file: str, transform: Optional[A.Compose] = None) -> None:
        super().__init__()

        try:
            self.df = pd.read_csv(csv_file)
        except FileNotFoundError("csv file not found.") as e:
            logger.exception(f"{e}")

        self.transform = transform

        logger.info(f"the number of samples: {len(self.df)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.df.iloc[idx]["img"]
        gt_path = self.df.iloc[idx]["gt"]

        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path)

        images = np.concatenate([img, gt], axis=2)

        if self.transform is not None:
            res = self.transform(image=images)['image']
            res = np.transpose(res, (2,0,1))
            img = res[0:3,:,:]
            gt = res[3:6,:,:]

        return img, gt, img_path
