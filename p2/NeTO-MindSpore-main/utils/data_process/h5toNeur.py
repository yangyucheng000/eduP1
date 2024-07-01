
import argparse
import numpy as np
import cv2
import h5py
from tqdm import trange
import imageio
from glob import glob
import trimesh
import os
from skimage import measure

def load_h5(h5_path, n_images):
    h5data = h5py.File(h5_path, 'r')
    print('loading data..............')
    projections = []
    screen_points = []
    masks  =[]
    light_mask = []
    views = np.array([0, 4, 8, 12, 16, 18, 20, 24, 28, 32, 36, 40, 44, 48, 52, 54, 56, 60, 64, 68])
    # for i in trange(n_images):
    for i in views:
        print(i)
        projections.append(h5data['cam_proj'][i])
        K = h5data['cam_k'][:]
        screen_pixel = h5data['screen_position'][i].reshape(resy, resx, -1)
        mask = h5data['mask'][i]

        blur = cv2.medianBlur(mask, 5)
        blur = cv2.bilateralFilter(blur, 10, 230, 230)
        ret, thresh1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = thresh1.copy()

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        blur = cv2.medianBlur(mask, 3)
        blur = cv2.bilateralFilter(blur, 8, 230, 230)
        ret, thresh1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = thresh1.copy()
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        valid = screen_pixel[:, :, 0] != 0
        light_mask.append(valid)
        masks.append(mask)
        screen_points.append(screen_pixel)

    return projections, K, masks, screen_points, light_mask

def load_h5_our(h5_path, n_images):
    h5data = h5py.File(h5_path, 'r')
    print('loading data..............')
    projections = []
    screen_points = []
    masks = []
    light_masks = []

    for i in range(n_images):
        projections.append(h5data['cam_proj'][i])
        K = h5data['cam_k'][:]
        screen_pixel = h5data['screen_position'][i].reshape(resy, resx, -1)
        mask = h5data['mask'][i].reshape(resy, resx)
        valid = screen_pixel[:, :, 0] != 0
        light_masks.append(valid)
        masks.append(mask)
        screen_points.append(screen_pixel)
        print(i)
    return projections, K, masks, screen_points, light_masks

def load_h5_(h5_path, n_images):
    h5data = h5py.File(h5_path, 'r')
    print('loading data..............')
    projections = []
    screen_points = []
    masks  =[]
    light_mask = []

    for i in trange(n_images):
        projections.append(h5data['cam_proj'][i])
        K = h5data['cam_k'][:]
        screen_pixel = h5data['screen_position'][i].reshape(resy, resx, -1)
        mask = h5data['mask'][i]

        blur = cv2.medianBlur(mask, 11)
        blur = cv2.bilateralFilter(blur, 30, 230, 230)
        ret, thresh1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = thresh1.copy()

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        blur = cv2.medianBlur(mask, 9)
        blur = cv2.bilateralFilter(blur, 10, 230, 230)
        ret, thresh1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = thresh1.copy()
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = mask.copy()
        valid = screen_pixel[:, :, 0] != 0
        light_mask.append(valid)
        masks.append(mask)
        screen_points.append(screen_pixel)

    return projections, K, masks, screen_points, light_mask



def get_scale_mat(mesh_path, hyperP=1.1):
    pcd = trimesh.load(mesh_path)

    vertices = pcd.vertices
    bbox_max = np.max(vertices, axis=0)
    bbox_min = np.min(vertices, axis=0)

    center = (bbox_max + bbox_min) * 0.5

    box = hyperP * (bbox_max - bbox_min).max() * 0.5
    scale_mat = np.diag([box, box, box, 1.0]).astype(np.float32)

    scale_mat[:3, 3] = center  # 缩放和平移
    return scale_mat

def compute_visualhull(masks, K, projections, mesh_path, box=200):
    resolution = 256

    minX, maxX = -box, box
    minY, maxY = -box, box
    minZ, maxZ = -box, box

    y, x, z = np.meshgrid(
        np.linspace(minX, maxX, resolution),
        np.linspace(minY, maxY, resolution),
        np.linspace(minZ, maxZ, resolution)
    )

    x = x[:, :, :, np.newaxis]
    y = y[:, :, :, np.newaxis]
    z = z[:, :, :, np.newaxis]
    coord = np.concatenate([x, y, z], axis=3)  # [res,res,res,3]

    volume = -np.ones(x.shape).squeeze()  # [res,res,res,1] ==>[res,res,res]

    f = K[0][0]
    cxp = K[0][2]
    cyp = K[1][2]
    for i in range(len(masks)):
        projection = projections[i]
        R = projection[:3, :3]
        T = projection[:3, 3]
        mask = masks[i]
        imgH = mask.shape[0]
        imgW = mask.shape[1]
        mask = mask.reshape(imgH * imgW)
        origin = -np.matmul(np.transpose(R), T)
        target = R[2, :] + origin
        up = -R[1, :]
        print("{}:{}".format(origin, target))
        yAxis = up / np.sqrt(np.sum(up * up))
        zAxis = target - origin
        zAxis = zAxis / np.sqrt(np.sum(zAxis * zAxis))
        xAxis = np.cross(zAxis, yAxis)
        xAxis = xAxis / np.sqrt(np.sum(xAxis * xAxis))
        Rot = np.stack([xAxis, yAxis, zAxis], axis=0)
        coordCam = np.matmul(Rot, np.expand_dims(coord - origin, axis=4))

        coordCam = coordCam.squeeze(4)
        xCam = coordCam[:, :, :, 0] / coordCam[:, :, :, 2]
        yCam = coordCam[:, :, :, 1] / coordCam[:, :, :, 2]
        #体素空间中每一个点的像素坐标
        xId = xCam * f + cxp
        yId = -yCam * f + cyp

        xInd = np.logical_and(xId >= 0, xId < imgW - 0.5)
        yInd = np.logical_and(yId >= 0, yId < imgH - 0.5)
        #体素空间中能投影到这个像素平面的3D点mask
        imInd = np.logical_and(xInd, yInd)

        xImId = np.round(xId[imInd]).astype(np.int32)
        yImId = np.round(yId[imInd]).astype(np.int32)

        maskInd = mask[yImId * imgW + xImId]

        volumeInd = imInd.copy()
        #能投影到像素平面上的3D点 将他的值变为mask中对应像素点的值
        volumeInd[imInd == 1] = maskInd
        #==0 表明这个3D点不能投影到mask 内
        volume[volumeInd == 0] = 1
        print('Occupied voxel: %d' % np.sum((volume > 0).astype(np.float32)))

        verts, faces, normals, _ = measure.marching_cubes_lewiner(volume, 0)
        #verts 在体素空间中的顶点 需要换成在世界坐标系下的距离
        print('Vertices Num: %d' % verts.shape[0])
        print('Normals Num: %d' % normals.shape[0])
        print('Faces Num: %d' % faces.shape[0])

        axisLen = float(resolution - 1) / 2.0
        #分辨率 顶点在真实世界中的3D坐标
        verts = (verts - axisLen) / axisLen * box
        mesh = trimesh.Trimesh(vertices=verts, vertex_normals=normals, faces=faces)
        # mesh.export(mesh_path)
    print('Export final mesh !')
    mesh.export(mesh_path)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/mnt/data4/Lzc/DRT_data')#/mnt/data/lzc/DRT_data  /mnt/data/lzc/DRT/data
    parser.add_argument('--mode', type=str, default='pig')
    parser.add_argument('--views', type=int, default=72)
    parser.add_argument('--save_path', type=str, default='/mnt/data/lzc/neus/drt')
    args = parser.parse_args()
    #首先导入mesh
    basedir = args.path
    name = args.mode
    n_images = args.views
    work_dir = args.save_path
    mesh_path = os.path.join(basedir,  name + '_vh.ply')
    scale_mat = get_scale_mat(mesh_path=mesh_path)
    resy = 1080   #960 1080 3472
    resx = 1920 # 1280 1920 4624
    h5_path = os.path.join(basedir,  name + '.h5')
    if args.mode in ['mouse', 'monkey', 'hand', 'dog']:
        w2c, K, masks, screen_points, light_masks = load_h5_(h5_path, n_images)
    elif args.mode in ['rabbit', 'pig', 'horse']:
        w2c, K, masks, screen_points, light_masks = load_h5(h5_path, n_images)
        compute_visualhull(masks, K, w2c, '/mnt/data4/Lzc/DRT_data/pig_20.ply', box=200)
    else:
        w2c, K, masks, screen_points, light_masks = load_h5_our(h5_path, n_images)
    print('final load data')

    compute_visualhull
    out_dir = os.path.join(work_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'light_mask'), exist_ok=True)

    for i, img in enumerate(masks):
        img = np.tile(img[:, :, None], (3))
        # img = img.rshape(resy, resx, -1)
        image_shape = img.shape
        imgs = np.zeros((image_shape[0] + 100, image_shape[1], image_shape[2]))
        imgs[:image_shape[0], :image_shape[1], :] = img#our data
        cv2.imwrite(os.path.join(out_dir, 'image', '{:0>3d}.png'.format(i)), imgs)
        cv2.imwrite(os.path.join(out_dir, 'mask', '{:0>3d}.png'.format(i)), imgs)

        light_mask = np.tile(light_masks[i][:, :, None], (3))
        imgs = np.zeros((image_shape[0] + 100, image_shape[1], image_shape[2]))
        imgs[:image_shape[0], :image_shape[1], :] = light_mask*255
        cv2.imwrite(os.path.join(out_dir, 'light_mask', '{:0>3d}.png'.format(i)), imgs)
    cam_dict = dict()
    intrinsic = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
    intrinsic[:3, :3] = K
    for i in range(n_images):
        world_mat = intrinsic @ w2c[i]
        world_mat = world_mat.astype(np.float32)
        cam_dict['camera_mat_{}'.format(i)] = intrinsic
        cam_dict['camera_mat_inv_{}'.format(i)] = np.linalg.inv(intrinsic)
        cam_dict['world_mat_{}'.format(i)] = world_mat
        cam_dict['world_mat_inv_{}'.format(i)] = np.linalg.inv(world_mat)

    if not os.path.exists(mesh_path):
        mesh_path = compute_visualhull(masks, K, w2c, mesh_path)

    print('final compute visual hull ')
    scale_mat = get_scale_mat(mesh_path=mesh_path)

    for i in range(n_images):
        cam_dict['scale_mat_{}'.format(i)] = scale_mat
        cam_dict['scale_mat_inv_{}'.format(i)] = np.linalg.inv(scale_mat)

    # 对3D点进行放缩
    scale_screen_points = []
    for i in range(len(screen_points)):
        screen_point = screen_points[i]
        screen_point = screen_point - scale_mat[:3, 3]
        screen_point = screen_point / scale_mat[0][0]
        s_shape = screen_point.shape
        tem = np.zeros((s_shape[0] + 100, s_shape[1], s_shape[2]))
        tem[:s_shape[0], :s_shape[1], :] = screen_point
        scale_screen_points.append(tem)

    pcd = trimesh.PointCloud(screen_point.reshape(-1, 3))  # [num,3]
    # work_dir = '/mnt/data/lzc/DRT_data/ray_point_'
    pcd.export(os.path.join(work_dir, name + 'ray_point.ply'))

    np.savez(os.path.join(out_dir, 'cameras_sphere.npz'), **cam_dict)
    np.save(os.path.join(out_dir, 'screen_point.npy') , np.array(scale_screen_points))