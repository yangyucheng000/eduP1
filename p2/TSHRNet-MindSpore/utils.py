import os
import random
from collections import OrderedDict
import matplotlib.pyplot as plt


def generate_training_data_list(training_data_dir, training_data_list_file):
    # shapenet_specular training dataset

    random.seed(1)

    path_i = []  # input
    path_a = []  # albedo
    path_s = []  # shading
    path_r = []  # specular residue
    path_d = []  # diffuse
    path_d_tc = []  # gamma correction version of diffuse
    path_m = []  # mask
    with open(training_data_list_file, 'r') as f:
        image_list = [x.strip() for x in f.readlines()]
    random.shuffle(image_list)
    for name in image_list:
        path_i.append(os.path.join(training_data_dir, name.split()[0]))  # input
        path_a.append(os.path.join(training_data_dir, name.split()[1]))  # albedo
        path_s.append(os.path.join(training_data_dir, name.split()[2]))  # shading
        path_r.append(os.path.join(training_data_dir, name.split()[3]))  # specular residue
        path_d.append(os.path.join(training_data_dir, name.split()[4]))  # diffuse
        path_d_tc.append(os.path.join(training_data_dir, name.split()[5]))  # gamma correction version of diffuse
        path_m.append(os.path.join(training_data_dir, name.split()[6]))  # mask

    num = len(image_list)
    path_i = path_i[:int(num)]
    path_a = path_a[:int(num)]
    path_s = path_s[:int(num)]
    path_r = path_r[:int(num)]
    path_d = path_d[:int(num)]
    path_d_tc = path_d_tc[:int(num)]
    path_m = path_m[:int(num)]

    path_list = {'path_i': path_i, 'path_a': path_a, 'path_s': path_s, 'path_r': path_r, 'path_d': path_d, 'path_d_tc': path_d_tc, 'path_m': path_m}
    return path_list


def generate_testing_data_list(data_dir, data_list_file):
    # shapenet_specular testing data

    path_i = []  # input
    path_a = []  # albedo
    path_s = []  # shading
    path_r = []  # specular residue
    path_d = []  # diffuse
    path_d_tc = []  # gamma correction version of diffuse
    path_m = []  # mask
    with open(data_list_file, 'r') as f:
        image_list = [x.strip() for x in f.readlines()]
    image_list.sort()
    for name in image_list:
        path_i.append(os.path.join(data_dir, name.split()[0]))  # input
        path_a.append(os.path.join(data_dir, name.split()[1]))  # albedo
        path_s.append(os.path.join(data_dir, name.split()[2]))  # shading
        path_r.append(os.path.join(data_dir, name.split()[3]))  # specular residue
        path_d.append(os.path.join(data_dir, name.split()[4]))  # diffuse
        path_d_tc.append(os.path.join(data_dir, name.split()[5]))  # gamma correction version of diffuse
        path_m.append(os.path.join(data_dir, name.split()[6]))  # mask

    num = len(image_list)
    path_i = path_i[:int(num)]
    path_a = path_a[:int(num)]
    path_s = path_s[:int(num)]
    path_r = path_r[:int(num)]
    path_d = path_d[:int(num)]
    path_d_tc = path_d_tc[:int(num)]
    path_m = path_m[:int(num)]

    path_list = {'path_i': path_i, 'path_a': path_a, 'path_s': path_s, 'path_r': path_r, 'path_d': path_d, 'path_d_tc': path_d_tc, 'path_m': path_m}

    return path_list


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict


def plot_log(data, dataset_name, save_model_name='model'):
    plt.cla()
    plt.plot(data['UNets'], label='L_total ')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.savefig('./logs_' + dataset_name + '/' + save_model_name + '.png')


def check_dir(dataset_name):
    if not os.path.exists('./logs_' + dataset_name):
        os.mkdir('./logs_' + dataset_name)
    if not os.path.exists('./checkpoints_' + dataset_name):
        os.mkdir('./checkpoints_' + dataset_name)
