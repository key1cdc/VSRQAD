import os
from shutil import copy


lr_path = '/media/vista/1E88DC0488DBD7FB/luzitao/dataset'
sr_path = '/home/vista/luZiTao/dataset/VSR/val/sr'
save_path = '/home/vista/luZiTao/dataset/VSR/val/lr'

sr_list = os.listdir(sr_path)
for sr in sr_list:
    video_, fps_, down_, scale_, method_, frame_ = sr.split('_')
    down_scale = 'LR' + scale_
    lr_img = video_ + '_' + fps_ + '_' + down_ + '_' + frame_
    lr_path_ = lr_path + '/' + video_ + '/' + down_scale + '/' + lr_img
    copy(lr_path_, save_path)
    print('ok')