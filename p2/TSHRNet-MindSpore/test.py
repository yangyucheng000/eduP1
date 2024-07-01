import argparse
import os
from collections import OrderedDict

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import ops
from tqdm import tqdm

from dataloader import ImageDataset
from UNet_mindspore_construct import UNet
from utils import generate_testing_data_list


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-l', '--load', type=str, default='60', help='the number of checkpoints')
    p.add_argument('-s', '--image_size', type=int, default=256)
    p.add_argument('-tedd', '--testing_data_dir', type=str, default='./test_dataset')
    p.add_argument('-tedlf', '--testing_data_list_file', type=str, default='./test_dataset/test_tuples.lst')
    p.add_argument('-mn', '--model_name', type=str, default='SSHR_ms')
    p.add_argument('-tdn', '--testing_data_name', type=str, default='SSHR')
    return p


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict


def unnormalize(x):
    x = x * 0.5 + 0.5
    return x


def test(UNet1, UNet2, UNet3, UNet4, model_name, test_dataset, test_dataloader):
    dir_refined = './test_result_' + model_name + '/' + key_dir + '/' + 'estimated_diffuse'
    dir_tc = './test_result_' + model_name + '/' + key_dir + '/' + 'estimated_diffuse_tc'

    for n, inputs in enumerate(tqdm(test_dataloader)):
        img, gt_albedo, gt_shading, gt_specular_residue, gt_diffuse, gt_diffuse_tc, object_mask = inputs

        # # estimations in our three-stage network
        # estimations in the first stage
        estimated_albedo = UNet1(img)
        estimated_shading = UNet2(img)
        estimated_specular_residue = (img - estimated_albedo * estimated_shading)

        # estimation in the second stage
        G3_input = ops.cat([estimated_albedo * estimated_shading, img], axis=1)
        estimated_diffuse_refined = UNet3(G3_input)

        # estimation in the third stage
        G4_input = ops.cat([estimated_diffuse_refined, estimated_specular_residue, img], axis=1)
        estimated_diffuse_tc = UNet4(G4_input)
        # # end

        temp = len(test_dataset.img_list['path_i'][n].split('/'))
        r_subdir = test_dataset.img_list['path_i'][n].split('/')[temp - 2]
        basename = test_dataset.img_list['path_i'][n].split('/')[temp - 1]
        subdir_refined = os.path.join(dir_refined, r_subdir)
        if not os.path.exists(subdir_refined):
            os.makedirs(subdir_refined)
        subdir_tc = os.path.join(dir_tc, r_subdir)
        if not os.path.exists(subdir_tc):
            os.makedirs(subdir_tc)

        estimated_diffuse_refined = (unnormalize(estimated_diffuse_refined)[0, :, :, :] * 255).numpy().transpose(1, 2, 0).astype('uint8')
        estimated_diffuse_refined = vision.ToPIL()(estimated_diffuse_refined)
        estimated_diffuse_name = os.path.join(subdir_refined, basename)
        estimated_diffuse_refined.save(estimated_diffuse_name)

        estimated_diffuse_tc = (unnormalize(estimated_diffuse_tc)[0, :, :, :] * 255).numpy().transpose(1, 2, 0).astype('uint8')
        # save tone-corrected diffuse image
        estimated_diffuse_tc = vision.ToPIL()(estimated_diffuse_tc)
        estimated_diffuse_tc_name = os.path.join(subdir_tc, basename)
        estimated_diffuse_tc.save(estimated_diffuse_tc_name)


def main(params):
    u_net1 = UNet(input_channels=3, output_channels=3)
    u_net2 = UNet(input_channels=3, output_channels=3)
    u_net3 = UNet(input_channels=6, output_channels=3)
    u_net4 = UNet(input_channels=9, output_channels=3)

    if params.load is not None:
        print('load checkpoint ' + params.load)
        # load UNet weights
        u_net1_weights = './checkpoints_' + params.model_name + '/UNet1_' + params.load + '.ckpt'
        u_net2_weights = './checkpoints_' + params.model_name + '/UNet2_' + params.load + '.ckpt'
        u_net3_weights = './checkpoints_' + params.model_name + '/UNet3_' + params.load + '.ckpt'
        u_net4_weights = './checkpoints_' + params.model_name + '/UNet4_' + params.load + '.ckpt'
        ms.load_checkpoint(u_net1_weights, u_net1)
        ms.load_checkpoint(u_net2_weights, u_net2)
        ms.load_checkpoint(u_net3_weights, u_net3)
        ms.load_checkpoint(u_net4_weights, u_net4)

    # size = params.image_size
    testing_data_dir = params.testing_data_dir
    testing_data_list_file = params.testing_data_list_file

    test_img_list = generate_testing_data_list(testing_data_dir, testing_data_list_file)
    test_dataset = ImageDataset(img_list=test_img_list, is_train=False)
    test_dataloader = ds.GeneratorDataset(test_dataset,
                                          column_names=['input', 'gt_albedo', 'gt_shading', 'gt_specular_residue',
                                                        'gt_diffuse', 'gt_diffuse_tc', 'object_mask'],
                                          shuffle=False, num_parallel_workers=1)
    test_dataloader = test_dataloader.batch(1)
    test(u_net1, u_net2, u_net3, u_net4, params.model_name, test_dataset, test_dataloader)


if __name__ == "__main__":
    parser = get_parser().parse_args()
    if parser.load is not None:
        load_num = str(parser.load)
        testing_data_name = parser.testing_data_name
        key_dir = testing_data_name + '_' + parser.model_name + '_' + load_num  # like SSHR_SSHR_60

    main(parser)
