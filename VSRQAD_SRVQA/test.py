import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from data import dataset
from option import args
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
from model.dream_SRVQA import dream_SRVQA
from utility import timer
import scipy.stats as stats
import pandas as pd
import cv2
from data import common

if len(args.gpu_ids) > 0:
    torch.cuda.set_device(args.gpu_ids[0])
    cudnn.benchmark = True

patch_size = 320
model = dream_SRVQA()
model.initialize(args)    # model parameter setting e.g. load pre-trained model
model.load_test_model()
model.set_mode(train=False)


# test_path 下有 SR，LR，ST
test_path = r'C:\Users\zitaolu\Desktop\VSR'
mos_path = r'C:\Users\zitaolu\Desktop\VSR\mos_with_names.csv'

SR_path = os.path.join(test_path, 'sr')
LR_path = os.path.join(test_path, 'lr')
ST_path = os.path.join(test_path, 'ST')



ave_loss = []
MOS = []
OS = []

images = os.listdir(SR_path)
images.sort()
with open(mos_path) as csv_file:
    datas = pd.read_csv(csv_file)
    for image in images:
        sr_image_path = SR_path + '/' + image
        name_, fps_, down_, scale_, method_, frame_ = image.split('_')
        lr_image_path = LR_path + '/' + name_ + '_' + fps_ + '_' + down_ + '_' + frame_
        video_name = name_ + '_' + fps_ + '_' + down_ + '_' + scale_ + '_' + method_ + '.mp4'
        ST_img = name_ + '_' + fps_ + '_' + down_ + '_' + scale_ + '_' + method_ + '.bmp'
        ST_img_path = ST_path + '/' + ST_img
        index = datas.loc[datas.name == video_name].index.to_list()
        mos = datas.iloc[index[0], 1]

        lr_ = cv2.imread(lr_image_path)
        sr_ = cv2.imread(sr_image_path)
        st_ = cv2.imread(ST_img_path)
        lr_ = cv2.cvtColor(lr_, cv2.COLOR_BGR2RGB)
        sr_ = cv2.cvtColor(sr_, cv2.COLOR_BGR2RGB)
        st_ = cv2.cvtColor(st_, cv2.COLOR_BGR2RGB)

        scale = sr_image_path.split('_')[-3]
        scale = list(scale)
        scale = int(scale[1])
        subim_sr, subim_lr = common.get_val_sr_lr_patch(sr_, lr_, patch_size, scale)
        # subim_sr = sr_
        # subim_lr = lr_
        subim_lr = cv2.resize(subim_lr, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
        subim_st = common.get_val_sr_patch(st_, patch_size)
        # subim_st = st_

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

        subim_sr, subim_lr, subim_st = common.np2Tensor([subim_sr, subim_lr, subim_st], 255)
        subim_sr = subim_sr.unsqueeze(0)
        subim_lr = subim_lr.unsqueeze(0)
        subim_st = subim_st.unsqueeze(0)

        data = {'sr': subim_sr, 'lr': subim_lr, 'st': subim_st}

        model.set_test_input(data)


        pre_mos = model.test()
        MOS.append(mos.item())
        OS.append(pre_mos.item())


srcc = stats.spearmanr(np.array(MOS), np.array(OS))
plcc = stats.pearsonr(np.array(MOS), np.array(OS))
rmse_sum = 0
for w in range(len(MOS)):
    rmse_sum += (MOS[w] - OS[w]) * (MOS[w] - OS[w])
rmse = np.sqrt(rmse_sum / len(MOS))

print('SRCC [{:.4f}] PLCC [{:.4f}] RMSE [{:.4f}]'.format(srcc.correlation,plcc[0],rmse))
