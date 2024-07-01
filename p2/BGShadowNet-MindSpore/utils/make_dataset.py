from typing import no_type_check
import numpy as np
from sklearn import mixture
from sklearn.cluster import KMeans
import cv2
import pandas as pd
import os


def get_average_color(x):
    b, g, r = x[:, 0], x[:, 1], x[:, 2]
    return np.array([np.mean(b), np.mean(g), np.mean(r)])

df = pd.DataFrame()
phase = 'val'
img_path ='/mnt/data/xxx/Dataset/xxx/'+phase+'/input/'#阴影图片
root_path ='/mnt/data/xxx/Dataset/xxx/'+phase+'/gt/'#无阴影图片
back_gt_apth ='/mnt/data/xxx/Dataset/xxx/'+phase+'/back_gt/'#真实的背景图片
paths = os.listdir(root_path)
paths.sort()
img_paths = []
gt_paths = []
back_gt_apths = []
background_colors = [[], [], []]
for path in paths:
    img_paths.append(img_path+path)
    gt_paths.append(root_path+path)
    back_gt_apths.append(back_gt_apth+path)
    # x = cv2.imread(root_path+path)
    # h, w, c = x.shape
    # x = x.flatten().reshape(h*w, c)
    # gmm = mixture.GaussianMixture(n_components=2, covariance_type='full')
    # gmm.fit(x)
    # #km = KMeans(n_clusters=2)
    # #km.fit(x)
    #
    # cls = gmm.predict(x.flatten().reshape(h*w, c))
    # #cls = km.predict(x.flatten().reshape(h*w, c))
    # cls0_colors = x[cls == 0]
    # cls1_colors = x[cls == 1]
    #
    # cls0_avg_color = get_average_color(cls0_colors)
    # cls1_avg_color = get_average_color(cls1_colors)
    #
    # print(cls0_avg_color)
    # #
    # print(cls1_avg_color)
    #
    #
    # if np.sum(cls0_avg_color)>=np.sum(cls1_avg_color):
    #     background_color = cls0_avg_color
    #     #cls = 1 - cls
    # else:
    #     background_color = cls1_avg_color
    #
    # gmm_out = np.array([cls0_avg_color if i == 0 else cls1_avg_color for i in cls])
    #cv2.imwrite('/home/heyinghao/code/some_shadow_remvoal_code/BEDSR-Net_A_Deep_Shadow_Removal_Network_from_a_Single_Document_Image-main/dataset/Jung/'+phase+'/gmm/gmm_{:s}.jpg'.format(path), gmm_out.reshape(h, w, c))
    #cv2.imwrite('./dataset/Jung/'+phase+'/kmeans/km_{:s}.jpg'.format(path), gmm_out.reshape(h, w, c))
    #cv2.imwrite('./dataset/Jung/'+phase+'/background/background_{:s}.jpg'.format(path), np.full_like(x, background_color).reshape(h, w, c))
    #cv2.imwrite('gmm/{:s}'.format(path), cls.reshape(h, w)*255)
    for i in range(3):
        background_colors[i].append(0)




df['img'] = img_paths
df['gt'] = gt_paths
df['back_gt'] = back_gt_apths
df['B'], df['G'], df['R'] = background_colors[0], background_colors[1], background_colors[2]

df.to_csv('../csv/xxx/'+phase+'.csv')

