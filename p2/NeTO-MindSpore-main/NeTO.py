import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms
from shutil import copyfile
# from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')

        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.uncertain_map = self.conf.get_bool('train.uncertain_map')

        self.views = self.conf.get_float('train.views', default=72)

        self.warm_up_end = self.conf.get_int('train.warm_up_end', default=0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.extIOR = self.conf.get_float('train.extIOR', default=1.0003)
        self.intIOR = self.conf.get_float('train.intIOR', default=1.4723)
        self.decay_rate = self.conf.get_float('train.decay_rate', default=0.1)
        self.n_samples = self.conf.get_int('model.neus_renderer.n_samples', default=0.1)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.refract_weight = self.conf.get_float('train.refract_weight')

        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.uncertain_masks = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf'])
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network'])
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network'])
        params_to_train += list(self.nerf_outside.get_parameters())
        params_to_train += list(self.sdf_network.get_parameters())
        params_to_train += list(self.deviation_network.get_parameters())
        decay_lr = self.dynamic_learning_rate()
        self.optimizer = nn.Adam(params_to_train, learning_rate=decay_lr)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     **self.conf['model.neus_renderer'])

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def initial_model(self, data):
        rays_o, rays_d, ray_point, mask, valid_mask = data[:, :3], data[:, 3: 6], \
                                                      data[:, 6: 9], data[:, 9: 10], data[:, 10:11][..., 0]
        near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
        background_rgb = None
        if self.use_white_bkgd:
            background_rgb = ops.ones([1, 3])

        if self.mask_weight > 0.0:
            mask = (mask > 0.5).float()
        else:
            mask = ops.ones_like(mask)

        mask_sum = mask.sum() + 1e-5

        render_out = self.renderer.render(rays_o, rays_d, near, far,
                                          background_rgb=background_rgb,
                                          cos_anneal_ratio=self.get_cos_anneal_ratio())

        gradient_error = render_out['gradient_error']
        weight_max = render_out['weight_max']
        weight_sum = render_out['weight_sum']
        s_val = render_out['s_val']
        cdf_fine = render_out['cdf_fine']

        eikonal_loss = gradient_error
        mask_loss = ops.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)
        loss = mask_loss * self.mask_weight + eikonal_loss * self.igr_weight
        return loss

    def detail_recontruction(self, data):
        rays_o, rays_d, ray_point, mask, valid_mask = data[:, :3], data[:, 3: 6], \
                                                      data[:, 6: 9], data[:, 9: 10], data[:, 10:11][..., 0]
        near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
        background_rgb = None
        valid_mask = valid_mask.bool()
        if self.use_white_bkgd:
            background_rgb = ops.ones([1, 3])

        if self.mask_weight > 0.0:
            mask = (mask > 0.5).float()
        else:
            mask = ops.ones_like(mask)
        mask_sum = mask.sum() + 1e-5
        render_out = self.renderer.render(rays_o, rays_d, near, far,
                                          background_rgb=background_rgb,
                                          cos_anneal_ratio=self.get_cos_anneal_ratio())
        gradient_error = render_out['gradient_error']
        weight_max = render_out['weight_max']
        weight_sum = render_out['weight_sum']
        s_val = render_out['s_val']
        cdf_fine = render_out['cdf_fine']
        normal_1 = render_out['gradients']
        inter_point = render_out['inter_point']

        l_t1, attenuate1, totalReflectMask1 = self.refraction(rays_d, normal_1,
                                                              eta1=1.0003, eta2=1.4723)
        rays_o = inter_point + l_t1 * 2
        rays_d = -l_t1
        near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
        render_out = self.renderer.render(rays_o, rays_d, near, far,
                                          background_rgb=background_rgb,
                                          cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                          path='bend')
        normal_2 = render_out['gradients']
        inter_point2 = render_out['inter_point']
        weight_sum2 = render_out['weight_sum']
        occlusion_mask = render_out['occlusion_mask']
        occlusion_sign = render_out['occlusion_sign']
        render_out_dir2, attenuate2, totalReflectMask2 = self.refraction(-rays_d, -normal_2,
                                                                         eta1=1.4723, eta2=1.0003)

        check_sign, check_sdf = self.check_sdf_val(inter_point, inter_point2)
        check_sdf = (check_sdf > 1e-3).sum(1)
        occlusion_mask = check_sdf == 0

        valid_mask = valid_mask & (~totalReflectMask1[:, 0]) & (~totalReflectMask2[:, 0]) & occlusion_mask

        inter_point2 = ops.stop_gradient(inter_point2)
        target = ray_point - inter_point2
        target = target / target.norm(dim=1, keepdim=True)
        diff = (render_out_dir2 - target)
        ray_loss = (diff[valid_mask]).pow(2).sum()
        eikonal_loss = gradient_error
        mask_loss = ops.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)
        loss = mask_loss * self.mask_weight + eikonal_loss * self.igr_weight + ray_loss * self.refract_weight

        return loss

    def check_sdf_val(self, intersection1, intersection2):
        first_second = (intersection2 - intersection1)
        rays_d = first_second / ops.norm(first_second, ord=2, dim=-1, keepdim=True)
        rays_o = intersection1

        z_vals = ops.linspace(0.0, 1.0, self.renderer.n_samples)

        check_z_vals = ops.norm(intersection2 - intersection1, ord=2, dim=-1, keepdim=True) * z_vals[None, :]

        pts = rays_o[:, None, :] + rays_d[:, None, :] * check_z_vals[..., :, None]
        check_sdf = self.sdf_network(pts.reshape(-1, 3)).reshape(-1, self.renderer.n_samples)
        check_sdf = ops.stop_gradient(check_sdf)

        for i in range(self.renderer.up_sample_steps // 2):
            new_check_z_vals = self.renderer.up_sample_occulsion(rays_o,
                                                                 rays_d,
                                                                 check_z_vals,
                                                                 check_sdf,
                                                                 self.renderer.n_importance // (
                                                                         self.renderer.up_sample_steps // 2),
                                                                 64 * 2 ** i)
            check_z_vals, check_sdf = self.renderer.cat_z_vals(rays_o,
                                                               rays_d,
                                                               check_z_vals,
                                                               new_check_z_vals,
                                                               check_sdf,
                                                               last=False)
        occlusion_sign = check_sdf.sign()
        occlusion_sign = ops.stop_gradient(occlusion_sign)
        check_sdf = ops.stop_gradient(check_sdf)
        return occlusion_sign, check_sdf

    def train(self, init_epoch):
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()
        print(image_perm)
        if self.uncertain_map:
            if self.iter_step % init_epoch == 0 and self.iter_step != 0:
                self.validate_mesh(world_space=False, resolution=512, threshold=args.mcube_threshold)
                self.compute_uncertain_map()
        # self.validate_image()
        for iter_i in tqdm(range(res_step)):
            indx = image_perm[self.iter_step % len(image_perm)]
            if self.iter_step >= init_epoch:
                """
                随机选择和uncertain map中选择
                """
                data, pixels_x, pixels_y = self.dataset.gen_ray_masks_near(indx, self.batch_size)
                grad_fn = ms.value_and_grad(self.detail_recontruction, None, self.optimizer.parameters, has_aux=False)
                loss, grad = grad_fn(data)
                self.optimizer(grad)
                self.iter_step += 1
            else:
                data, pixels_x, pixels_y = self.dataset.gen_random_rays_at(indx, self.batch_size)
                grad_fn = ms.value_and_grad(self.initial_model, None, self.optimizer.parameters, has_aux=False)
                loss, grad = grad_fn(data)
                self.optimizer(grad)
                self.iter_step += 1

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {}'.format(self.iter_step, loss))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def refraction(self, l, normal, eta1, eta2):
        cos_theta = ops.sum(l * (-normal), dim=1).unsqueeze(1)  # [10, 1, 192, 256] 鏍囬噺
        i_p = l + normal * cos_theta
        t_p = eta1 / eta2 * i_p

        t_p_norm = ops.sum(t_p * t_p, dim=1)

        totalReflectMask = (ops.stop_gradient(t_p_norm) > 0.999999).unsqueeze(1)

        t_i = ops.sqrt(1 - ops.clamp(t_p_norm, 0, 0.999999)).unsqueeze(1).expand_as(normal) * (-normal)
        t = t_i + t_p
        t = t / ops.sqrt(ops.clamp(ops.sum(t * t, dim=1), min=1e-10)).unsqueeze(1)

        cos_theta_t = ops.sum(t * (-normal), dim=1).unsqueeze(1)

        e_i = (cos_theta_t * eta2 - cos_theta * eta1) / \
              ops.clamp(cos_theta_t * eta2 + cos_theta * eta1, min=1e-10)
        e_p = (cos_theta_t * eta1 - cos_theta * eta2) / \
              ops.clamp(cos_theta_t * eta1 + cos_theta * eta2, min=1e-10)

        attenuate = ops.clamp(0.5 * (e_i * e_i + e_p * e_p), 0, 1)
        attenuate = ops.stop_gradient(attenuate)

        return t, attenuate, totalReflectMask

    def reflection(self, l, normal):
        cos_theta = ops.sum(l * (-normal), dim=1).unsqueeze(1)
        r_p = l + normal * cos_theta
        r_p_norm = ops.clamp(ops.sum(r_p * r_p, dim=1), 0, 0.999999)
        r_i = ops.sqrt(1 - r_p_norm).unsqueeze(1).expand_as(normal) * normal
        r = r_p + r_i
        r = r / ops.sqrt(ops.clamp(ops.sum(r * r, dim=1), min=1e-10).unsqueeze(1))

        return r

    def get_image_perm(self):
        if self.views == self.dataset.n_images:
            randperm = ops.Randperm(max_length=self.dataset.n_images, pad=-1)
            n = ms.Tensor([self.dataset.n_images], dtype=ms.int32)
            output = randperm(n)
            return output
        elif self.dataset.n_images % self.views == 0:
            return ops.linspace(0, self.dataset.n_images - 1, self.dataset.n_images)[::
                                                                                     int(self.dataset.n_images // self.views)].int()
        elif self.views == 20:
            return ms.Tensor([0, 4, 8, 12, 16, 18, 20, 24, 28, 32, 36, 40, 44, 48, 52, 54, 56, 60, 64, 68])

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def dynamic_learning_rate(self):
        lr = []
        for i in range(self.warm_up_end):
            learning_factor = i / self.warm_up_end
            lr.append(self.learning_rate * learning_factor)
        for i in range(self.warm_up_end, self.end_iter):
            alpha = self.learning_rate_alpha
            progress = (i - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
            lr.append(self.learning_rate * learning_factor)
        return lr

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self):
        nerf_param = ms.load_checkpoint(os.path.join(self.base_exp_dir, 'checkpoints', 'nerf_400000.ckpt'))
        param_not_load, _ = ms.load_param_into_net(self.nerf_outside, nerf_param)
        sdf_param = ms.load_checkpoint(
            os.path.join(self.base_exp_dir, 'checkpoints', 'sdf_network_400000.ckpt'))
        param_not_load, _ = ms.load_param_into_net(self.sdf_network, sdf_param)
        variance_param = ms.load_checkpoint(
            os.path.join(self.base_exp_dir, 'checkpoints', 'variance_network_400000.ckpt'))
        param_not_load, _ = ms.load_param_into_net(self.deviation_network, variance_param)
        self.iter_step = 400000

    def save_checkpoint(self):
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        ms.save_checkpoint(self.nerf_outside, os.path.join(self.base_exp_dir, 'checkpoints',
                                                           'nerf_{:0>6d}'.format(self.iter_step)))
        ms.save_checkpoint(self.sdf_network, os.path.join(self.base_exp_dir, 'checkpoints',
                                                          'sdf_network_{:0>6d}'.format(self.iter_step)))
        ms.save_checkpoint(self.deviation_network, os.path.join(self.base_exp_dir, 'checkpoints',
                                                                'variance_network_{:0>6d}'.format(
                                                                    self.iter_step)))

    def validate_mesh(self, world_space=False, resolution=256, threshold=0.0):
        bound_min = ms.Tensor(self.dataset.object_bbox_min, dtype=ms.float32)
        bound_max = ms.Tensor(self.dataset.object_bbox_max, dtype=ms.float32)

        vertices, triangles = \
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        print(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')


if __name__ == '__main__':
    print('Hello ZC')

    ms.set_context(device_target='GPU')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base_9.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--init_epoch', type=int, default=50001)
    args = parser.parse_args()

    ms.set_context(device_id=args.gpu)
    # ms.set_context(mode=ms.PYNATIVE_MODE, pynative_synchronize=True)
    ms.set_context(mode=ms.PYNATIVE_MODE)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train(args.init_epoch)
    elif args.mode == 'test':
        runner.load_checkpoint()
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)

