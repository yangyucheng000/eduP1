from PIL import Image

from data_transform import ImageTransform


class ImageDataset:
    def __init__(self, img_list, is_train=True):
        """init the class object to hold the data"""
        self.img_list = img_list
        self.img_trans = ImageTransform(is_train=is_train)

    def __getitem__(self, index):
        """overrode the getitem method to support random access"""
        input_img = Image.open(self.img_list['path_i'][index]).convert('RGB')
        gt_albedo = Image.open(self.img_list['path_a'][index]).convert('RGB')
        gt_shading = Image.open(self.img_list['path_s'][index]).convert('RGB')
        gt_specular_residue = Image.open(self.img_list['path_r'][index]).convert('RGB')
        gt_diffuse = Image.open(self.img_list['path_d'][index]).convert('RGB')
        gt_diffuse_tc = Image.open(self.img_list['path_d_tc'][index]).convert('RGB')
        object_mask = Image.open(self.img_list['path_m'][index]).convert('RGB')
        input_img = self.img_trans(input_img)
        gt_albedo = self.img_trans(gt_albedo)
        gt_shading = self.img_trans(gt_shading)
        gt_specular_residue = self.img_trans(gt_specular_residue)
        gt_diffuse = self.img_trans(gt_diffuse)
        gt_diffuse_tc = self.img_trans(gt_diffuse_tc)
        object_mask = self.img_trans(object_mask)
        return input_img, gt_albedo, gt_shading, gt_specular_residue, gt_diffuse, gt_diffuse_tc, object_mask

    def __len__(self):
        """specify the length of data"""
        return len(self.img_list['path_i'])
