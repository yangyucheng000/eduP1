import os
import argparse
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import trimesh

def nn_correspondance(verts1, verts2):
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    distances = distances.reshape(-1)

    return distances

def eval_hand(mesh_pred, mesh_trgt, threshold=.05, down_sample=.02):
    pcd_pred = o3d.geometry.PointCloud(mesh_pred.vertices)
    pcd_trgt = o3d.geometry.PointCloud(mesh_trgt.vertices)
    down_sample = .02
    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)
    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)
    verts_pred = verts_pred[verts_pred[:, 1] > 20]
    verts_trgt = verts_trgt[verts_trgt[:, 1] > 20]
    dist1 = nn_correspondance(verts_pred, verts_trgt)  #
    dist2 = nn_correspondance(verts_trgt, verts_pred)
    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))  #
    fscore = 2 * precision * recal / (precision + recal)
    metrics = {
        'Acc': np.mean(dist2),
        'Comp': np.mean(dist1),
        'Prec': precision,
        'Recal': recal,
        'F-score': fscore,
    }
    return metrics


def evaluate(mesh_pred, mesh_trgt, threshold=.05, down_sample=.02):

    pcd_pred = o3d.geometry.PointCloud(mesh_pred.vertices)
    pcd_trgt = o3d.geometry.PointCloud(mesh_trgt.vertices)

    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    dist1 = nn_correspondance(verts_pred, verts_trgt) #
    dist2 = nn_correspondance(verts_trgt, verts_pred)

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))#
    fscore = 2 * precision * recal / (precision + recal)
    metrics = {
        'Acc': np.mean(dist2),
        'Comp': np.mean(dist1),
        'Prec': precision,
        'Recal': recal,
        'F-score': fscore,
    }
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Li', type=str, default='./confs/base.conf')
    parser.add_argument('--DRT', type=str, default='/mnt/data4/Lzc/res/20views/DRT')
    parser.add_argument('--Our', type=str, default='/mnt/data4/Lzc/DRT_data/DRT_neus/exp')
    parser.add_argument('--gt', type=str, default='/mnt/data4/Lzc/res')
    args = parser.parse_args()

    models = ['horse', 'tiger', 'rabbit', 'pig', 'hand', 'dog', 'mouse', 'monkey']
    for i in range(len(models)):
        model = models[i]
        print('=============================' + model + '================')
        # model = 'monkey'
        target = os.path.join(args.gt, model + '_scan.ply')
        source = os.path.join(args.DRT, model + '400.080.002_20_recons.ply')
        pcd = trimesh.load(target)

        vertices = pcd.vertices
        bbox_max = np.max(vertices, axis=0)
        bbox_min = np.min(vertices, axis=0)
        threshold = (bbox_max - bbox_min).max() / 256
        print('=============================DRT================')
        print(evaluate(o3d.io.read_triangle_mesh(source),
                       o3d.io.read_triangle_mesh(target), threshold=threshold))
        source = os.path.join(args.Our, model, 'final_20', 'meshes',  '00290000.ply')
        print("=====================================our===============================")
        print(evaluate(o3d.io.read_triangle_mesh(source),
                       o3d.io.read_triangle_mesh(target), threshold=threshold))
        print("=====================================our===============================")
        source = os.path.join(args.Li, model + '_pred_scale.ply')
        target = os.path.join(args.Li, model + '_gt_scale.ply')
        print(evaluate(o3d.io.read_triangle_mesh(source),
                       o3d.io.read_triangle_mesh(target), threshold=threshold))


parser = argparse.ArgumentParser()
parser.add_argument('--Li', type=str, default='/mnt/data4/Lzc/res/9views/20')
parser.add_argument('--DRT', type=str, default='/mnt/data4/Lzc/res/9views')
parser.add_argument('--Our', type=str, default='/mnt/data4/Lzc/DRT_data/DRT_neus/exp')
parser.add_argument('--gt', type=str, default='/mnt/data4/Lzc/res')
args = parser.parse_args()

models = ['pig', 'mouse', 'monkey', 'dog', 'horse', 'tiger','hand', 'rabbit']
for i in range(len(models)):
    model = models[i]
    print('=============================' + model + '================')
    # model = 'monkey'
    target = os.path.join(args.gt, model + '_scan.ply')
    source = os.path.join(args.DRT, model + '_400.080.002_9_recons.ply')
    pcd = trimesh.load(target)

    vertices = pcd.vertices
    bbox_max = np.max(vertices, axis=0)
    bbox_min = np.min(vertices, axis=0)
    threshold = (bbox_max - bbox_min).max() / 256
    print('=============================DRT================')
    print(evaluate(o3d.io.read_triangle_mesh(source),
                   o3d.io.read_triangle_mesh(target), threshold=threshold))
    source = os.path.join(args.Our, model, 'final_9', 'meshes', '00300000.ply')
    print("=====================================our===============================")
    print(evaluate(o3d.io.read_triangle_mesh(source),
                   o3d.io.read_triangle_mesh(target), threshold=threshold))

    print("=====================================li===============================")
    source = os.path.join(args.Li, model + '_pred_scale.ply')
    target = os.path.join(args.Li, model + '_gt_scale.ply')
    print(evaluate(o3d.io.read_triangle_mesh(source),
                   o3d.io.read_triangle_mesh(target), threshold=threshold))

