

import os
import cv2
import numpy as np




# val_lr = '/home/vista/luZiTao/dataset/CVPR_competition/video_resolution/test/test_sharp_bicubic/X4/'
# results_dir = '/home/vista/luZiTao/python_code/compettion/NTIRE2021/bicubic_results'
#
# for idx in range(0,30):
#     print('No.{} filename'.format(idx))
#     filename = '{:0>3}'.format(idx)
#     for ii in range(0,100):
#         input_img_list = []
#         target_img_list = []
#         m = ii
#         mm = '{:0>8}'.format(m)
#         bicubic_filename = filename + '_' + mm
#         target_path = val_lr + filename + '/' + str(mm) + '.npy'
#         target_ = np.load(target_path, allow_pickle=False)
#         target_ = cv2.cvtColor(target_, cv2.COLOR_BGR2RGB)
#         dst2 = cv2.resize(target_,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
#         save_name = os.path.join(results_dir, '{}.png'.format(bicubic_filename))
#         cv2.imwrite(save_name,dst2)
#         print('ok')

#npy
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
val_lr =r'D:\tecent_data\video007_frame\train'
# results_dir = '/home/vista/luZiTao/dataset/CVPR_DIV2K/train/train_sharp_bicubic/X4'
for root, _, names in os.walk(val_lr):
    for name in names:
        if is_image_file(name):
            target_name = os.path.join(root, name)
            target_img = cv2.imread(target_name)
            target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
            save_name = target_name[:-3] + 'npy'

            np.save(save_name, target_img)

#xiacaiyang
# for ii in range(1,801):
#     input_img_list = []
#     target_img_list = []
#     m = ii
#     mm = '{:0>4}'.format(m)
#     bicubic_filename =mm
#     target_path = val_lr + '/' + str(mm) + '.npy'
#     target_ = np.load(target_path, allow_pickle=False)
#     target_ = cv2.cvtColor(target_, cv2.COLOR_BGR2RGB)
#     dst2 = cv2.resize(target_,None,fx=0.25,fy=0.25,interpolation=cv2.INTER_CUBIC)
#     save_name = os.path.join(results_dir, '{}.png'.format(bicubic_filename))
#     cv2.imwrite(save_name,dst2)
#     print('ok')

