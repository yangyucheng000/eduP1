import argparse
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import time
from logging import DEBUG, INFO, basicConfig, getLogger
from libs.models.tiramisu import *
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import wandb
from albumentations import (
    Compose,
    RandomResizedCrop,
    Rotate,
    HorizontalFlip,
    VerticalFlip,
    Transpose,
    ColorJitter,
    CoarseDropout,
    Normalize,
    Affine,
)
# from albumentations.pytorch import ToTensorV2

from libs.checkpoint import resume, save_checkpoint
from libs.config import get_config
from libs.dataset import get_dataloader
# from libs.device import get_device
from libs.helper import evaluate, train
from libs.logger import TrainLogger
from libs.loss_fn import get_criterion
from libs.models import get_model
from libs.seed import set_seed

logger = getLogger(__name__)


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments.
    解析来自命令行界面的所有参数 返回一个被解析的参数列表被解析的参数"""

    #  1.定义一个ArgumentParser实例:
    parser = argparse.ArgumentParser(
        description="""
        train a network for image classification with Flowers Recognition Dataset.
        """
    )
    #  2.往参数对象添加参数，str的config，help是功能介绍； action - 命令行遇到参数时的动作，默认值是 store。
    #   store_true 是指带触发action时为真，不触发则为假。通俗讲是指运行程序是否带参数,
    parser.add_argument("--config", type=str, default="./configs/model=benet/config.yaml", help="path of a config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Add --resume option if you start training from checkpoint.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Add --use_wandb option if you want to use wandb.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Add --debug option if you want to see debug-level logs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    # 3. # 获取并返回所有参数
    return parser.parse_args()

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs：
    将学习率设置为初始LR，每2个 epochs衰减10。"""
    # lr =lr/2**(epoch//100)
    lr = lr*(0.5**(epoch//30))  # 这意味着每30个历时的学习率将减少0.5。
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main() -> None:
    args = get_arguments()

    # save log files in the directory which contains config file.
    mindspore.set_context(device_target='GPU')
    result_path = os.path.dirname(args.config)
    experiment_name = os.path.basename(result_path)

    # setting logger configuration
    logname = os.path.join(result_path, f"{datetime.datetime.now():%Y-%m-%d}_train.log")

    basicConfig(
        level=DEBUG if args.debug else INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=logname,
    )

    # fix seed
    set_seed()

    # configuration
    config = get_config(args.config)

    # Dataloader 图像数据预处理
    train_transform = Compose(
        [
            RandomResizedCrop(config.height, config.width),
            HorizontalFlip(),
            Normalize(mean=(0.5, ), std=(0.5, )),
        ]
    )

    train_loader = get_dataloader(
        config.dataset_name,
        config.model,
        "train",
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
        transform=train_transform,
    )

    # define a model
    model = FCDenseNet57(3)     # TODO add model initializer
    optimizer = nn.Adam(model.trainable_params(), learning_rate=config.learning_rate)

    # keep training and validation log
    begin_epoch = 0
    best_loss = float("inf")

    # resume if you want
    if args.resume:
        resume_path = os.path.join(result_path, "checkpoint.pth")
        begin_epoch, model, optimizer, best_loss = resume(resume_path, model, optimizer)

    log_path = os.path.join(result_path, "log.csv")
    train_logger = TrainLogger(log_path, resume=args.resume)

    # criterion for loss
    criterion = get_criterion(config.loss_function_name)

    # Weights and biases
    if args.use_wandb:
        wandb.init(
            name=experiment_name,
            config=config,
            project="benet",
            job_type="training",
            #dirs="./wandb_result/",
        )
        # Magic
        wandb.watch(model, log="all")

    # train and validate model
    logger.info("Start training.")

    for epoch in range(begin_epoch, config.max_epoch):
        # training
        start = time.time()
        train_loss = train(
            train_loader, model, criterion, optimizer, epoch       # device
        )
        train_time = int(time.time() - start)

        # validation
        start = time.time()
        val_loss = 1
        val_time = int(time.time() - start)

        # save a model if top1 acc is higher than ever
        # 因为使用自己的数据集没有严格划分验证集，所以保存在训练集上效果最好的参数
        if best_loss > train_loss:
            best_loss = train_loss
            print("current best loss:{}, epoch:{}".format(train_loss, epoch))
            mindspore.save_checkpoint(
                model,
                os.path.join(result_path, "pretrained_benet.ckpt"),
            )

        # save checkpoint every epoch
        save_checkpoint(result_path, epoch, model, optimizer, best_loss)

        # write logs to dataframe and csv file
        train_logger.update(
            epoch,
            optimizer.get_lr(),
            train_time,
            train_loss,
            val_time,
            val_loss,
        )

        # save logs to wandb
        if args.use_wandb:
            wandb.log(
                {
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_time[sec]": train_time,
                    "train_loss": train_loss,
                    "val_time[sec]": val_time,
                    "val_loss": val_loss,
                },
                step=epoch,
            )

    # save models
    mindspore.save_checkpoint(model, os.path.join(result_path, "checkpoint.ckpt"))      # .state_dict()

    logger.info("Done")


if __name__ == "__main__":
    main()
