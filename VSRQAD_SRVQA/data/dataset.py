import os
import torch.utils.data as data
import cv2
import numpy as np
from data.matlab_imresize import imresize
from data import common
import torch
import pandas as pd


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_npy_file(filename):
    return filename.endswith('.npy')

def make_dataset(img_path, sr_factor, neigh_num=5, stage='train', npy=False):
    """
        img_path : input image path
        sr_factor : 2/3/4 for SR or 1 for CAR
        stage :  train/val/test
    """
    img_path_list = []
    img_path_ = os.path.join(img_path, stage)
    if stage == 'train':
        # hr_path = os.path.join(img_path_, 'hr')
        sr_path = os.path.join(img_path_, 'sr')
        lr_path = os.path.join(img_path_, 'lr')
        ST_path = os.path.join(img_path, 'ST')
        images = os.listdir(sr_path)
        mos_path = os.path.join(img_path_, 'mos_with_names.csv')
        with open(mos_path) as csv_file:
            datas = pd.read_csv(csv_file)
            for image in images:
                sr_image_path = sr_path + '/' + image
                name_, fps_, down_, scale_, method_, frame_ = image.split('_')
                lr_image_path = lr_path + '/' + name_ + '_' + fps_ + '_' + down_ + '_' + frame_
                video_name = name_ + '_' + fps_ + '_' + down_ + '_' + scale_ + '_' + method_ + '.mp4'
                ST_img = name_ + '_' + fps_ + '_' + down_ + '_' + scale_ + '_' + method_ + '.bmp'
                ST_img_path = ST_path + '/' + ST_img
                index = datas.loc[datas.name == video_name].index.to_list()
                mos = datas.iloc[index[0], 1]
                img_path_list.append({'sr': sr_image_path, 'lr': lr_image_path, 'st': ST_img_path, 'mos': mos})
                # print('ok')

    elif stage == 'val':
        # hr_path = os.path.join(img_path_, 'hr')
        sr_path = os.path.join(img_path_, 'sr')
        lr_path = os.path.join(img_path_, 'lr')
        ST_path = os.path.join(img_path, 'ST')
        images = os.listdir(sr_path)
        mos_path = os.path.join(img_path_, 'mos_with_names.csv')
        with open(mos_path) as csv_file:
            datas = pd.read_csv(csv_file)
            for image in images:
                sr_image_path = sr_path + '/' + image
                name_, fps_, down_, scale_, method_, frame_ = image.split('_')
                lr_image_path = lr_path + '/' + name_ + '_' + fps_ + '_' + down_ + '_' + frame_
                video_name = name_ + '_' + fps_ + '_' + down_ + '_' + scale_ + '_' + method_ + '.mp4'
                ST_img = name_ + '_' + fps_ + '_' + down_ + '_' + scale_ + '_' + method_ + '.bmp'
                ST_img_path = ST_path + '/' + ST_img
                index = datas.loc[datas.name == video_name].index.to_list()
          
                mos = datas.iloc[index[0], 1]
                img_path_list.append({'sr': sr_image_path, 'lr': lr_image_path, 'st': ST_img_path, 'mos': mos})
    print('ok')
    return img_path_list


def save_npy(img_path, stage='train'):
    img_path = os.path.join(img_path, stage)
    print('Saving .npy file from images.')
    for root, _, names in os.walk(img_path):
        for name in names:
            if is_image_file(name):
                target_name = os.path.join(root, name)
                target_img = cv2.imread(target_name)
                target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
                save_name = target_name[:-3] + 'npy'
                np.save(save_name, target_img)


class DatasetTrain(data.Dataset):
    def __init__(self, args):
        super(DatasetTrain, self).__init__()
        self.scale_param = args.scale_param
        self.img_path = args.train_root
        self.npy = args.npy
        self.upscale = args.upscale
        self.frames_num = args.frames_num
        if args.data_reset and self.npy:
            save_npy(self.img_path)
        self.train_path = make_dataset(self.img_path, self.scale_param, self.frames_num, 'train', self.npy)
        self.patch_size = args.patch_size
        self.rgb_range = args.rgb_range
        self.augment = args.augment
        self.n_colors = args.n_colors


    def __getitem__(self, idx):
        path = self.train_path[idx]
        sr_path = path['sr']
        lr_path = path['lr']
        st_path = path['st']
        mos = path['mos']

        lr_ = cv2.imread(lr_path)
        sr_ = cv2.imread(sr_path)
        st_ = cv2.imread(st_path)
        lr_ = cv2.cvtColor(lr_, cv2.COLOR_BGR2RGB)
        sr_ = cv2.cvtColor(sr_, cv2.COLOR_BGR2RGB)
        st_ = cv2.cvtColor(st_, cv2.COLOR_BGR2RGB)
        scale = sr_path.split('_')[-3]
        scale = list(scale)
        scale = int(scale[1])
        subim_sr, subim_lr = common.get_sr_lr_patch(sr_, lr_, self.patch_size, scale)
        subim_st = common.get_sr_patch(st_,self.patch_size)
        subim_lr = cv2.resize(subim_lr, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
        subim_sr = subim_sr / 255.0
        R, G, B = cv2.split(subim_sr)
        R = (R - 0.485) / 0.229
        G = (G - 0.456) / 0.224
        B = (B - 0.406) / 0.225
        subim_sr = cv2.merge([R, G, B])

        subim_st = subim_st / 255.0
        R, G, B = cv2.split(subim_st)
        R = (R - 0.485) / 0.229
        G = (G - 0.456) / 0.224
        B = (B - 0.406) / 0.225
        subim_st = cv2.merge([R, G, B])

        subim_lr = subim_lr / 255.0
        R, G, B = cv2.split(subim_lr)
        R = (R - 0.485) / 0.229
        G = (G - 0.456) / 0.224
        B = (B - 0.406) / 0.225
        subim_lr = cv2.merge([R, G, B])

        subim_sr, subim_lr = common.hr_lr_augmentation(subim_sr, subim_lr)
        subim_st = common.sr_augmentation(subim_st)
        subim_sr, subim_lr, subim_st = common.np2Tensor([subim_sr, subim_lr, subim_st], self.rgb_range)

        images = {'sr': subim_sr, 'lr': subim_lr, 'st': subim_st, 'mos': mos, 'scale': scale}
        return images

    def __len__(self):
        return len(self.train_path)
        # return 40


class DatasetTest(data.Dataset):
    def __init__(self, args):
        super(DatasetTest, self).__init__()
        self.augment = args.augment
        self.scale_param = args.scale_param
        self.img_path = args.test_root
        self.stage = args.stage
        if self.stage == 'train':
            self.stage = 'val'
        self.npy = args.npy
        # if args.data_reset and self.npy:
        if args.data_reset and self.npy:
            save_npy(self.img_path)
        self.frames_num = args.frames_num
        self.test_path = make_dataset(self.img_path, self.scale_param, self.frames_num, self.stage, self.npy)
        self.patch_size = args.patch_size
        self.rgb_range = args.rgb_range
        self.upscale = args.upscale
        self.n_colors = args.n_colors

    def __getitem__(self, idx):
        path = self.test_path[idx]
        sr_path = path['sr']
        lr_path = path['lr']
        st_path = path['st']
        mos = path['mos']

        lr_ = cv2.imread(lr_path)
        sr_ = cv2.imread(sr_path)
        st_ = cv2.imread(st_path)
        lr_ = cv2.cvtColor(lr_, cv2.COLOR_BGR2RGB)
        sr_ = cv2.cvtColor(sr_, cv2.COLOR_BGR2RGB)
        st_ = cv2.cvtColor(st_, cv2.COLOR_BGR2RGB)

        scale = sr_path.split('_')[-3]
        scale = list(scale)
        scale = int(scale[1])
        subim_sr, subim_lr = common.get_val_sr_lr_patch(sr_, lr_, self.patch_size, scale)
        subim_lr = cv2.resize(subim_lr, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
        subim_st = common.get_val_sr_patch(st_, self.patch_size)

        subim_sr = subim_sr / 255.0
        R, G, B = cv2.split(subim_sr)
        R = (R - 0.485) / 0.229
        G = (G - 0.456) / 0.224
        B = (B - 0.406) / 0.225
        subim_sr = cv2.merge([R, G, B])

        subim_st = subim_st / 255.0
        R, G, B = cv2.split(subim_st)
        R = (R - 0.485) / 0.229
        G = (G - 0.456) / 0.224
        B = (B - 0.406) / 0.225
        subim_st = cv2.merge([R, G, B])

        subim_lr = subim_lr / 255.0
        R, G, B = cv2.split(subim_lr)
        R = (R - 0.485) / 0.229
        G = (G - 0.456) / 0.224
        B = (B - 0.406) / 0.225
        subim_lr = cv2.merge([R, G, B])

        subim_sr, subim_lr, subim_st = common.np2Tensor([subim_sr, subim_lr, subim_st], self.rgb_range)

        images = {'sr': subim_sr, 'lr': subim_lr, 'st': subim_st, 'mos': mos, 'scale':scale}
        return images

    def __len__(self):
        return len(self.test_path)      #120


if __name__ == '__main__':
    img_path = '/media/luo/data/data/NTIRE2021/video_resolution'
    make_dataset(img_path, 1, 5, 'train')
