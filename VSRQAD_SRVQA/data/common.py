
import random

import cv2
import numpy as np
import skimage.color as sc
import torch
import imageio

def save_image(img_tensor, save_name):
    img_tensor = img_tensor.squeeze(0)
    img_np = tensor2Np([img_tensor], 255)

    # img_np[0] = np.clip(img_np[0],0,255)
    imageio.imsave(save_name, img_np[0])
    print('Saving image into {}'.format(save_name))

def get_vqa_patch(img_lr, img_hr, img_sr, patch_size, scale):
    ih, iw = img_lr[0].shape[:2]

    scale = img_sr[0].shape[0]//img_lr[0].shape[0]
    p = scale
    tp = patch_size
    ip = tp // scale
    img_lr_list = []
    img_hr_list = []
    img_sr_list = []
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy
    for i in range(5):
        img_lr_ = img_lr[i][iy:iy + ip, ix:ix + ip, :]
        img_hr_ = img_hr[i][ty:ty + tp, tx:tx + tp, :]
        img_sr_ = img_sr[i][ty:ty + tp, tx:tx + tp, :]
        img_lr_list.append(img_lr_)
        img_hr_list.append(img_hr_)
        img_sr_list.append(img_sr_)

    return img_lr_list, img_hr_list, img_sr_list

def get_hr_sr_patch(img_hr, img_sr, patch_size):
    ih, iw = img_sr.shape[:2]
    ip = patch_size
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    img_hr_ = img_hr[iy:iy + ip, ix:ix + ip, :]
    img_sr_ = img_sr[iy:iy + ip, ix:ix + ip, :]
    return img_hr_, img_sr_

def get_sr_patch(img_sr, patch_size):
    ih, iw = img_sr.shape[:2]
    ip = patch_size
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    img_sr_ = img_sr[iy:iy + ip, ix:ix + ip, :]
    return img_sr_

def get_sr_lr_patch(img_sr, img_lr, patch_size, scale):
    ih, iw = img_lr.shape[:2]
    ip = patch_size//scale
    p = scale
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    img_lr_ = img_lr[iy:iy + ip, ix:ix + ip, :]
    img_sr_ = img_sr[(iy*p):(iy*p+patch_size), (ix*p):(ix*p+patch_size), :]
    return img_sr_, img_lr_

def get_val_sr_lr_patch(img_sr, img_lr, patch_size, scale):
    ip = patch_size//scale
    p = scale
    img_lr_ = img_lr[50:50 + ip, 50:50 + ip, :]
    img_sr_ = img_sr[(50*p):(50*p+patch_size), (50*p):(50*p+patch_size), :]
    return img_sr_, img_lr_

def get_val_hr_sr_patch(img_hr, img_sr, patch_size):
    # ih, iw = img_sr.shape[:2]
    # ip = patch_size
    # ix = random.randrange(0, iw - ip + 1)
    # iy = random.randrange(0, ih - ip + 1)
    img_hr_ = img_hr[100:(100+patch_size), 100:(100+patch_size), :]
    img_sr_ = img_sr[100:(100+patch_size), 100:(100+patch_size), :]
    return img_hr_, img_sr_

def get_val_sr_patch(img_sr, patch_size):
    # ih, iw = img_sr.shape[:2]
    # ip = patch_size
    # ix = random.randrange(0, iw - ip + 1)
    # iy = random.randrange(0, ih - ip + 1)
    img_sr_ = img_sr[300:(300+patch_size), 500:(500+patch_size), :]
    return img_sr_


def get_vqa_sr_patch(img_lr, img_hr, img_sr, patch_size, scale):
    ih, iw = img_lr[0].shape[:2]
    p = scale
    tp = patch_size
    ip = tp // scale
    img_lr_list = []
    img_hr_list = []
    img_sr_list = []
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = ix, iy
    for i in range(5):
        # img_lr_ = img_lr[i][ty:ty + tp, tx:tx + tp, :]
        img_lr_ = img_lr[i]
        img_hr_ = img_hr[i][ty:ty + tp, tx:tx + tp, :]
        img_sr_ = img_sr[i][ty:ty + tp, tx:tx + tp, :]
        img_lr_list.append(img_lr_)
        img_hr_list.append(img_hr_)
        img_sr_list.append(img_sr_)

    return img_lr_list, img_hr_list, img_sr_list

def get_patch(img_in, img_tar, patch_size, scale):
    ih, iw = img_in[0].shape[:2]
    p = scale
    tp = p * patch_size
    ip = tp // scale
    img_in_list=[]
    img_tar_list=[]
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy
    for i in range(5):
        img_in_ = img_in[i][iy:iy + ip, ix:ix + ip, :]
        img_tar_ = img_tar[i][ty:ty + tp, tx:tx + tp, :]
        img_in_list.append(img_in_)
        img_tar_list.append(img_tar_)
    return img_in_list, img_tar_list

def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channel == 1 and c == 3:    #change rgb to y
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:   #change y to yyy
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]


def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # make code run faster
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(
            rgb_range / 255)  # if you want rgb_range 255, then tensor div 1; elif you want rgb_range 1; then tensor div 255.

        return tensor

    return [_np2Tensor(_l) for _l in l]

def tensor2Np(l, rgb_range):
    def _tensor2Np(img):
        if not isinstance(img, torch.Tensor):
            raise TypeError('Input is not tensor')
        elif len(img.size()) != 3:
            raise ValueError('input size must be C*H*W')

        tensor = img.data.mul(255 / rgb_range).round()
        img_np = tensor.byte().permute(1, 2, 0).cpu().numpy()

        return img_np

    return [_tensor2Np(_l) for _l in l]


# def add_noise(x, noise='.'):   # noise is a list ['G', 0.5] or ['S', 5]
#     if noise is not '.':
#         noise_type = noise[0]
#         noise_value = int(noise[1:])
#         if noise_type == 'G':
#             noises = np.random.normal(scale=noise_value, size=x.shape)
#             noises = noises.round()
#         elif noise_type == 'P':
#             noises = np.random.poisson(x * noise_value) / noise_value
#             noises = noises - noises.mean(axis=0).mean(axis=0)
#
#         x_noise = x.astype(np.int16) + noises.astype(np.int16)
#         x_noise = x_noise.clip(0, 255).astype(np.uint8)
#         return x_noise
#     else:
#         return x

def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(_l) for _l in l]

def upscale(lr_list, scale):
    out_list = []
    dst_w = lr_list[0].shape[0]
    dst_h = lr_list[0].shape[1]

    for lr in lr_list:
        out_list.append(cv2.resize(lr, (dst_h * scale, dst_w * scale), interpolation=cv2.INTER_LINEAR))
    return out_list

def hr_down_up_scale(img_hr, scale):
    ih, iw = img_hr.shape[:2]
    down_img = cv2.resize(img_hr, (iw//scale, ih//scale), interpolation=cv2.INTER_CUBIC)
    up_img = cv2.resize(down_img, (iw, ih), interpolation=cv2.INTER_LINEAR)
    return up_img

def hr_lr_augmentation(img_hr, img_lr):
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot90 = random.random() < 0.5
    if hflip:
        img_lr = img_lr[:, ::-1, :]
        img_hr = img_hr[:, ::-1, :]
    if vflip:
        img_lr = img_lr[::-1, :, :]
        img_hr = img_hr[::-1, :, :]
    if rot90:
        img_lr = img_lr.transpose(1, 0, 2)
        img_hr = img_hr.transpose(1, 0, 2)
    return img_hr, img_lr

def sr_augmentation(img_sr):
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot90 = random.random() < 0.5
    if hflip:
        img_sr = img_sr[:, ::-1, :]
    if vflip:
        img_sr = img_sr[::-1, :, :]
    if rot90:
        img_sr = img_sr.transpose(1, 0, 2)
    return img_sr

def augmentation_vqa(lr_list, hr_list, sr_list):
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot90 = random.random() < 0.5
    lr_out_list=[]
    hr_out_list=[]
    sr_out_list=[]
    for i in range(5):
        lr_img = lr_list[i]
        hr_img = hr_list[i]
        sr_img = sr_list[i]
        if hflip:
            lr_img = lr_list[i][:, ::-1, :]
            hr_img = hr_list[i][:, ::-1, :]
            sr_img = sr_list[i][:, ::-1, :]
        if vflip:
            lr_img = lr_list[i][::-1, :, :]
            hr_img = hr_list[i][::-1, :, :]
            sr_img = sr_list[i][::-1, :, :]
        if rot90:
            lr_img = lr_list[i].transpose(1, 0, 2)
            hr_img = hr_list[i].transpose(1, 0, 2)
            sr_img = sr_list[i].transpose(1, 0, 2)
        lr_out_list.append(lr_img)
        hr_out_list.append(hr_img)
        sr_out_list.append(sr_img)
    return lr_out_list, hr_out_list, sr_out_list

def augmentation(in_list,tar_list):
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot90 = random.random() < 0.5
    input_list=[]
    target_list=[]
    for i in range(5):
        in_img = in_list[i]
        tar_img = tar_list[i]
        if hflip:
            in_img = in_list[i][:, ::-1, :]
            tar_img = tar_list[i][:, ::-1, :]
        if vflip:
            in_img = in_list[i][::-1, :, :]
            tar_img = tar_list[i][::-1, :, :]
        if rot90:
            in_img = in_list[i].transpose(1, 0, 2)
            tar_img = tar_list[i].transpose(1, 0, 2)
        input_list.append(in_img)
        target_list.append(tar_img)
    return input_list,target_list
def modcrop(imgs, modulo):

    if len(imgs.shape) == 2 or imgs.shape[2] == 1:
        sz = imgs.shape
        sz = sz - np.mod(sz, modulo)
        imgs = imgs[:sz[0], :sz[1]]
    else:
        tmpsz = imgs.shape
        sz = tmpsz[:2]
        sz = sz - np.mod(sz, modulo)
        imgs = imgs[:sz[0], :sz[1], :]

    return imgs



def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
