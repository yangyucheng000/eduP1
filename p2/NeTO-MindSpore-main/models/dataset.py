import mindspore as ms
import mindspore.ops as ops
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        # self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')
        self.screen_point_name = conf.get_string('screen_point_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))

        self.screen_point_np = np.load(os.path.join(self.data_dir, self.screen_point_name))# 72 h*w 3

        self.camera_dict = camera_dict


        self.masks_lis = sorted(glob_imgs(os.path.join(self.data_dir, 'mask')))
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0
        self.masks_np = self.masks_np[:,:,:, :1]>0.8

        self.n_images = len(self.masks_lis)

        mask_bound = []
        for i in range(len(self.masks_np)):#[299:1079, 259:1654]
            pixely, pixelx, _ = np.where(self.masks_np[i])
            mask_bound.append([pixely.min(), pixely.max(), pixelx.min(), pixelx.max()])
        self.masks_bound_np = np.stack(np.array(mask_bound))


        self.light_mask_lis = sorted(glob_imgs(os.path.join(self.data_dir, 'light_mask')))
        self.light_mask_np = np.stack([cv.imread(im_name) for im_name in self.light_mask_lis]) / 256.0
        self.light_mask_np = self.light_mask_np[:, :, :, :1] > 0.8


        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        self.B, self.H, self.W = self.masks_np.shape[0], self.masks_np.shape[1], self.masks_np.shape[2]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(ms.Tensor(intrinsics).float())
            self.pose_all.append(ms.Tensor(pose).float())

        self.light_masks = ms.Tensor(self.light_mask_np.astype(np.float32)) # [n_images, H, W, 3]
        self.masks = ms.Tensor(self.masks_np.astype(np.float32))   # [n_images, H, W, 3]
        self.masks_bound = ms.Tensor(self.masks_bound_np.astype(np.float32))
        self.intrinsics_all = ops.stack(self.intrinsics_all)   # [n_images, 4, 4]
        self.intrinsics_all_inv = ops.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = ops.stack(self.pose_all)  # [n_images, 4, 4]

        self.image_pixels = self.H * self.W
        self.screen_point = ms.Tensor(self.screen_point_np.astype(np.float32))#[n_image, H*W, 3]

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = ops.randint(low=0, high=self.W, size=(batch_size, ))
        pixels_y = ops.randint(low=0, high=self.H, size=(batch_size, ))
        mask = self.masks[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        p = ops.stack([pixels_x, pixels_y, ops.ones_like(pixels_y)], axis=-1).float()  # batch_size, 3
        p = ops.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / ops.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = ops.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3] # batch_size, 3
        rays_o = rays_o.broadcast_to(rays_v.shape)
        # load screen point

        ray_point = self.screen_point[img_idx][(pixels_y, pixels_x)]
        valid_mask = self.light_masks[img_idx][(pixels_y, pixels_x)]  #[bacthsize,1]

        return ops.cat([rays_o, rays_v, ray_point, mask, valid_mask], axis=-1), pixels_x, pixels_y

        # batch_size, 10

    def gen_random_rays_at_mask(self, img_idx, uncertain_map, batch_size):
        """
        Generate mask image rays at world space from one camera.
        """
        pixels_y, pixels_x  = ops.where(uncertain_map)# torch.where return {H, W }
        num = (uncertain_map).sum()
        index = ops.randint(low=0, high=num, size=(batch_size, ), dtype=ms.int32)
        pixels_x, pixels_y = pixels_x[index], pixels_y[index]

        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        p = ops.stack([pixels_x, pixels_y, ops.ones_like(pixels_y)], axis=-1).float()  # batch_size, 3
        p = ops.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / ops.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = ops.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].broadcast_to(rays_v.shape)  # batch_size, 3
        # load screen point

        ray_point = self.screen_point[img_idx][(pixels_y, pixels_x)]
        valid_mask = self.light_masks[img_idx][(pixels_y, pixels_x)]  # [bacthsize,1]

        return ops.cat([rays_o, rays_v, ray_point, mask, valid_mask], axis=-1), pixels_x, pixels_y

    def gen_random_rays_at_mix(self, img_idx, uncertain_map, batch_size):
        data1, pixels_x1, pixels_y1 = self.gen_ray_masks_near(img_idx, batch_size // 7*6)
        data2, pixels_x2, pixels_y2 = self.gen_random_rays_at_mask(img_idx, uncertain_map, batch_size // 7*1)
        # ray_o  ray_d ray_point mask valid_mask
        return ops.vstack((data1, data2)), ops.vstack((pixels_x1.unsqueeze(1), pixels_x2.unsqueeze(1)))[..., 0], \
               ops.vstack((pixels_y1.unsqueeze(1), pixels_y2.unsqueeze(1)))[..., 0]


    def gen_ray_masks_near(self, img_idx, batch_size):
        pixels_y = ops.randint(low=np.max([self.masks_bound_np[img_idx][0]-100, 0]).item(),
                               high=np.min([self.masks_bound_np[img_idx][1]+100, self.H]).item(),
                               size=(batch_size, ), dtype=ms.int32)#heigh
        pixels_x = ops.randint(low=np.max([self.masks_bound_np[img_idx][2]-100, 0]).item(),
                               high=np.min([self.masks_bound_np[img_idx][3]+100, self.W]).item(),
                               size=(batch_size, ), dtype=ms.int32)#wifth

        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        p = ops.stack([pixels_x, pixels_y, ops.ones_like(pixels_y)], axis=-1).float()  # batch_size, 3
        p = ops.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / ops.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = ops.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].broadcast_to(rays_v.shape)  # batch_size, 3
        # load screen point
        # index = pixels_y * self.W + pixels_x

        ray_point = self.screen_point[img_idx][(pixels_y, pixels_x)]
        valid_mask = self.light_masks[img_idx][(pixels_y, pixels_x)]  # [bacthsize,1]

        return ops.cat([rays_o, rays_v, ray_point, mask, valid_mask], axis=-1), pixels_x, pixels_y

    def gen_ray_at_mask(self, img_idx, batch_size):
        pixels_y, pixels_x, _ = ops.where(self.light_masks[img_idx] > 0.9)
        num = (self.light_masks[img_idx] > 0.9).sum()
        index = ops.randint(low=0, high=num, size=(batch_size))
        pixels_x, pixels_y = pixels_x[index], pixels_y[index]

        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        p = ops.stack([pixels_x, pixels_y, ops.ones_like(pixels_y)], axis=-1).float()  # batch_size, 3
        p = ops.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / ops.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = ops.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].broadcast_to(rays_v.shape)  # batch_size, 3
        # load screen point
        ray_point = self.screen_point[img_idx][(pixels_y, pixels_x)]
        valid_mask = self.light_masks[img_idx][(pixels_y, pixels_x)]  # [bacthsize,1]
        return ops.cat([rays_o, rays_v, ray_point, mask, valid_mask], axis=-1), pixels_x, pixels_y

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = ops.linspace(0, self.W - 1, self.W // l)
        ty = ops.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = ops.meshgrid(tx, ty)
        p = ops.stack([pixels_x, pixels_y, ops.ones_like(pixels_y)], axis=-1)  # W, H, 3
        p = ops.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / ops.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].asnumpy()
        pose_0 = ops.stop_gradient(pose_0)
        pose_1 = self.pose_all[idx_1].asnumpy()
        pose_1 = ops.stop_gradient(pose_1)
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = ms.Tensor(pose[:3, :3])
        trans = ms.Tensor(pose[:3, 3])
        rays_v = ops.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].broadcast_to(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_rays_at_ray_point(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = ops.linspace(0, self.W - 1, self.W // l)
        ty = ops.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = ops.meshgrid(tx, ty)
        p = ops.stack([pixels_x, pixels_y, ops.ones_like(pixels_y)], axis=-1)  # W, H, 3
        p = ops.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / ops.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = ops.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].broad_cast_to(rays_v.shape)  # W, H, 3

        ray_point = self.screen_point[img_idx]  # [H, W, 3]
        valid_mask = self.light_masks[img_idx]  # [H, W, 1]
        mask = self.masks[img_idx]
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1), ray_point, valid_mask, mask

    def near_far_from_sphere(self, rays_o, rays_d):
        a = ops.sum(rays_d ** 2, dim=-1, keepdim=True)
        b = 2.0 * ops.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.masks_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)