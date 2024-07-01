import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import datetime
from libs.fix_weight_dict import fix_model_state_dict
import time
from logging import DEBUG, INFO, basicConfig, getLogger
import wandb    # 训练可视化工具
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from albumentations import (#数据增强工具
    Compose,
    RandomResizedCrop,
    Resize,
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
from libs.models.tiramisu import *
from libs.models.tiramisuskip import *
from libs.models.STLtiramisuskip import *
from libs.models.models import Discriminator
from libs.checkpoint import resume_BEDSRNet, save_checkpoint_BEDSRNet
from libs.config import get_config
from libs.dataset import get_dataloader
# from libs.device import get_device
from libs.helper_bedsrnet import evaluate, train
from libs.logger import TrainLoggerBEDSRNet
from libs.loss_fn import get_criterion
from libs.models import get_model
from libs.seed import set_seed

logger = getLogger(__name__)


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="""
        train a network for image classification with Flowers Recognition Dataset.
        """
    )
    parser.add_argument("--config", type=str, help="path of a config file", default="configs/model=bedsrnet/config.yaml")
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

    return parser.parse_args()


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    #lr =lr/2**(epoch//100)
    if epoch>200:
        lr = lr*(0.7**((epoch-150)//30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class WarmUpLR():
# class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def main() -> None:
    args = get_arguments()

    # save log files in the directory which contains config file.
    result_path = os.path.join(os.path.dirname(args.config), 'checkpoint')
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

    # cpu or cuda
    mindspore.set_context(device_target='GPU', device_id=0)

    # Dataloader
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

    # the number of classes
    n_classes = 1

    # define a model
    benet = FCDenseNet57(3)                 # 背景估计网络
    benet_weights = mindspore.load_checkpoint('pretrained/ms_trained_model/pretrained_benet.ckpt')
    fix_model_state_dict(benet_weights)
    mindspore.load_param_into_net(benet, benet_weights)

    generator = FCDenseSkipNet57(3)         # 第一阶段网络
    refine_net = STLFCDenseSkipNet57(6)      # 第二阶段网络
    discriminator = Discriminator(6)        # 鉴别器
    if config.pretrained == True:
        generator_weights = mindspore.load_checkpoint('pretrained/generator_ms.ckpt')
        fix_model_state_dict(generator_weights)
        mindspore.load_param_into_net(generator, generator_weights)

        discriminator_weights = mindspore.load_checkpoint('pretrained/discriminator_ms.ckpt')
        fix_model_state_dict(discriminator_weights)
        mindspore.load_param_into_net(discriminator, discriminator_weights)

        refine_weights = mindspore.load_checkpoint('pretrained/refine_net_ms_3.ckpt')
        fix_model_state_dict(refine_weights)
        mindspore.load_param_into_net(refine_net, refine_weights)

    polynomial_decay_lr = nn.PolynomialDecayLR(learning_rate=config.learning_rate, end_learning_rate=0.000001,
                                               decay_steps=40, power=1.0, update_decay_steps=True)
    optimizerG = nn.Adam(refine_net.trainable_params(), learning_rate=polynomial_decay_lr, beta1=config.beta1, beta2=config.beta2)
    optimizerD = nn.Adam(discriminator.trainable_params(), learning_rate=polynomial_decay_lr, beta1=config.beta1, beta2=config.beta2)
    optimizerGen = nn.Adam(generator.trainable_params(), learning_rate=polynomial_decay_lr, beta1=config.beta1, beta2=config.beta2)
    lambda_dict = {"lambda1": config.lambda1, "lambda2": config.lambda2}
    warmup_epoch = 4
    iter_per_epoch = 4371 // config.batch_size
    # warmup_scheduler = nn.warmup_lr(optimizerG, iter_per_epoch * warmup_epoch, 1, warmup_epoch)

    # keep training and validation log
    begin_epoch = 0
    best_g_loss = float("inf")
    best_d_loss = float("inf")

    # resume if you want
    # if args.resume:
    #     resume_path = config.resume_path
    #     begin_epoch, generator, discriminator, optimizerG, optimizerD, optimizerGen, best_g_loss, best_d_loss = resume_BEDSRNet(
    #         resume_path, generator, discriminator, optimizerG, optimizerD, optimizerGen)

    log_path = os.path.join(result_path, "log.csv")
    train_logger = TrainLoggerBEDSRNet(log_path, resume=args.resume)

    # criterion for loss
    criterion = get_criterion(config.loss_function_name)

    # Weights and biases
    if args.use_wandb:
        wandb.init(
            name=experiment_name,
            config=config,
            project="bedsrnet",
            job_type="training",
            #dirs="./wandb_result/",
        )
        # Magic
        #wandb.watch(model, log="all")
        wandb.watch(generator, log="all")
        wandb.watch(discriminator, log="all")

    # train and validate model
    logger.info("Start training.")

    for epoch in range(begin_epoch, config.max_epoch):
        # training
        start = time.time()
        train_g_loss, train_d_loss = train(
            train_loader, generator,refine_net, discriminator,benet, criterion, lambda_dict,
            optimizerG, optimizerD, optimizerGen, epoch # , device
        )
        train_time = int(time.time() - start)
        print("epoch: {}, train_g_loss: {}, train_d_loss: {}".format(epoch, train_g_loss, train_d_loss))
        # if epoch >= warmup_epoch:
        #     adjust_learning_rate(optimizerG, epoch, config.learning_rate)
        #     optimizerG.get_parameters()
        # print('learn rate', optimizerG.param_groups[0]['lr'])

        # validation
        start = time.time()
        val_g_loss, val_d_loss = 1.0, 1.0
        val_time = int(time.time() - start)
        
        if epoch%20==0 and epoch>200:
            mindspore.save_checkpoint(discriminator,
                                      os.path.join(result_path, "pretrained_discriminator_latest.ckpt"))
            mindspore.save_checkpoint(generator,
                                      os.path.join(result_path, "pretrained_generator_latest.ckpt"))
            mindspore.save_checkpoint(refine_net,
                                      os.path.join(result_path, "pretrained_refine_latest.ckpt"))
        # save a model if top1 acc is higher than ever
        if best_g_loss > train_g_loss:
            print("current best loss:{}, epoch:{}".format(train_g_loss, epoch))
            best_g_loss = train_g_loss
            best_d_loss = train_d_loss
            mindspore.save_checkpoint(
                generator,
                os.path.join(result_path, "pretrained_generator.ckpt"),
            )
            mindspore.save_checkpoint(
                refine_net,
                os.path.join(result_path, "pretrained_refine_net.ckpt"),
            )
            mindspore.save_checkpoint(
                discriminator,
                os.path.join(result_path, "pretrained_discriminator.ckpt"),
            )

        # save checkpoint every epoch
        save_checkpoint_BEDSRNet(result_path, epoch, generator, discriminator, optimizerG, optimizerD, optimizerGen,
                                 best_g_loss, best_d_loss)

        # write logs to dataframe and csv file
        train_logger.update(
            epoch,
            optimizerG.get_lr(),
            optimizerD.get_lr(),
            train_time,
            train_g_loss,
            train_d_loss,
            val_time,
            val_g_loss,
            val_d_loss
        )

        # save logs to wandb
        if args.use_wandb:
            wandb.log(
                {
                    "lrG": optimizerG.get_lr(),
                    "lrD": optimizerD.get_lr(),
                    "train_time[sec]": train_time,
                    "train_g_loss": train_g_loss,
                    "train_d_loss": train_d_loss,
                    "val_time[sec]": val_time,
                    "val_g_loss": val_g_loss,
                    "val_d_loss": val_d_loss,
                },
                step=epoch,
            )

    logger.info("Done")


if __name__ == "__main__":
    main()