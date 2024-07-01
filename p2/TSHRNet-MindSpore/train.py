import argparse

import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, value_and_grad, context
from mindspore.ops import cat
from tqdm import tqdm

from UNet_mindspore_construct import UNet
from dataloader import ImageDataset
from utils import generate_training_data_list, plot_log, check_dir

ms.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ne', '--num_epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('-bs', '--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('-lne', '--load_num_epoch', type=str, default=None, help='the number of checkpoints')
    parser.add_argument('-s', '--image_size', type=int, default=256)
    parser.add_argument('-cs', '--crop_size', type=int, default=256)
    parser.add_argument('-lr', '--lr', type=float, default=1e-4)
    parser.add_argument('-pdn', '--pretrained_dataset_name', type=str, default='SSHR_ms', help='pretrained model name')
    # settings for training and testing data, which can be from different datasets
    # Here, using testing data to generate some temp results for observing variations in the results
    # for "dataset_name" (e.g. SSHR_SSHR), the first and second "SSHR"s refer to training and testing dataset name, respectively
    # this "dataset_name" can be used for mkdir specific dirs for saving results
    parser.add_argument('-dn', '--dataset_name', type=str, default='SSHR')
    parser.add_argument('-trdd', '--train_data_dir', type=str, default='G:/SSHR/')
    parser.add_argument('-trdlf', '--train_data_list_file', type=str, default='G:/SSHR/SSHR/train_7_tuples.lst')
    return parser


def train_model(u_net1, u_net2, u_net3, u_net4, dataloader, load_num_epoch, num_epoch, lr, dataset_name,
                save_model_name='model'):
    check_dir(dataset_name)

    cell_list = nn.CellList()
    cell_list.append(u_net1)
    cell_list.append(u_net2)
    cell_list.append(u_net3)
    cell_list.append(u_net4)

    loss_criterion = nn.MSELoss()
    optimizer_u_net = nn.Adam([{'params': cell_list[0].trainable_params()},
                               {'params': cell_list[1].trainable_params()},
                               {'params': cell_list[2].trainable_params()},
                               {'params': cell_list[3].trainable_params()}],
                              learning_rate=lr, beta1=0.5, beta2=0.999)

    batch_size = dataloader.batch_size

    u_nets_loss = []
    if load_num_epoch is not None:
        epoch_old = int(load_num_epoch)
    else:
        epoch_old = 0

    def forward_fn(input_forward):
        img, gt_albedo, gt_shading, gt_specular_residue, gt_diffuse, gt_diffuse_tc, object_mask = input_forward
        # # estimations in our three-stage network
        # estimations in the first stage
        estimated_albedo = cell_list[0](img)
        estimated_shading = cell_list[1](img)
        estimated_specular_residue = (img - estimated_albedo * estimated_shading)

        # estimation in the second stage
        g3_input = cat([estimated_albedo * estimated_shading, img], axis=1)
        estimated_diffuse_refined = cell_list[2](g3_input)

        # estimation in the third stage
        g4_input = cat([estimated_diffuse_refined, estimated_specular_residue, img], axis=1)
        estimated_diffuse_tc = cell_list[3](g4_input)

        # loss for the first stage (physics-based specular removal)
        l_albedo = loss_criterion(estimated_albedo, gt_albedo)
        l_shading = loss_criterion(estimated_shading, gt_shading)
        l_specular_residue = loss_criterion(estimated_specular_residue, gt_specular_residue)

        # loss for the second stage (specular-free refinement)
        l_diffuse_refined = loss_criterion(estimated_diffuse_refined, gt_diffuse)

        # loss for the thrid stage (tone correction)
        l_diffuse_tc = loss_criterion(estimated_diffuse_tc, gt_diffuse_tc)

        # # end
        l_total = l_albedo + l_shading + l_specular_residue + l_diffuse_refined + l_diffuse_tc

        return l_total

    def train_step(input_train):
        losses, grads = value_and_grad(forward_fn, None, optimizer_u_net.parameters)(input_train)  # get values and gradients
        optimizer_u_net(grads)  # update gradient
        return losses

    for epoch in range(1, num_epoch + 1):
        epoch = epoch + epoch_old
        epoch_l_total = 0.0

        tqdm.write(f'Epoch {epoch}/{num_epoch + epoch_old}')

        for inputs in tqdm(dataloader):
            loss = train_step(inputs)
            epoch_l_total += loss

        tqdm.write('epoch {} || Epoch_Net_Loss:{:.4f}'.format(epoch, float(epoch_l_total) / batch_size))

        u_nets_loss += [epoch_l_total / batch_size]
        plot_log({'UNets': u_nets_loss}, dataset_name, save_model_name)

        if epoch % 10 == 0:
            ms.save_checkpoint(cell_list[0], 'checkpoints_' + dataset_name + '/' + save_model_name + '1_' + str(epoch) + 'ckpt')
            ms.save_checkpoint(cell_list[1], 'checkpoints_' + dataset_name + '/' + save_model_name + '2_' + str(epoch) + 'ckpt')
            ms.save_checkpoint(cell_list[2], 'checkpoints_' + dataset_name + '/' + save_model_name + '3_' + str(epoch) + 'ckpt')
            ms.save_checkpoint(cell_list[3], 'checkpoints_' + dataset_name + '/' + save_model_name + '4_' + str(epoch) + 'ckpt')

            # update learning rate
            lr /= 10

    return cell_list


def main(parser):
    u_net1 = UNet(input_channels=3, output_channels=3)
    u_net2 = UNet(input_channels=3, output_channels=3)
    u_net3 = UNet(input_channels=6, output_channels=3)
    u_net4 = UNet(input_channels=9, output_channels=3)

    batch_size = parser.batch_size
    num_epoch = parser.num_epoch
    train_data_dir = parser.train_data_dir
    train_data_list_file = parser.train_data_list_file
    lr = parser.lr
    dataset_name = parser.dataset_name
    pretrained_dataset_name = parser.pretrained_dataset_name
    load_num_epoch = parser.load_num_epoch

    if parser.load_num_epoch is not None:
        print('load checkpoint ' + parser.load_num_epoch)
        # load UNet weights
        u_net1_weights = './checkpoints_' + pretrained_dataset_name + '/UNet1_' + parser.load_num_epoch + '.ckpt'
        u_net2_weights = './checkpoints_' + pretrained_dataset_name + '/UNet2_' + parser.load_num_epoch + '.ckpt'
        u_net3_weights = './checkpoints_' + pretrained_dataset_name + '/UNet3_' + parser.load_num_epoch + '.ckpt'
        u_net4_weights = './checkpoints_' + pretrained_dataset_name + '/UNet4_' + parser.load_num_epoch + '.ckpt'
        ms.load_checkpoint(u_net1_weights, u_net1)
        ms.load_checkpoint(u_net2_weights, u_net2)
        ms.load_checkpoint(u_net3_weights, u_net3)
        ms.load_checkpoint(u_net4_weights, u_net4)

    train_img_list = generate_training_data_list(train_data_dir, train_data_list_file)
    train_dataset = ImageDataset(img_list=train_img_list)

    train_dataloader = ds.GeneratorDataset(train_dataset,
                                           column_names=['input', 'gt_albedo', 'gt_shading', 'gt_specular_residue', 'gt_diffuse', 'gt_diffuse_tc', 'object_mask'],
                                           shuffle=True, num_parallel_workers=4)
    train_dataloader = train_dataloader.batch(batch_size)

    train_model(u_net1, u_net2, u_net3, u_net4, dataloader=train_dataloader, load_num_epoch=load_num_epoch,
                num_epoch=num_epoch, lr=lr, dataset_name=dataset_name, save_model_name='UNet')


if __name__ == "__main__":
    main(get_parser().parse_args())
