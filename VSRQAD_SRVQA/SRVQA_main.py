
import torch
from data import dataset
from option import args
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
# from visualizer import Visualizer
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
from model.srfusemodel import SRFuseModel
from model.basemodel import tensor2Np
from utility import timer
import matplotlib.pyplot as plt


torch.manual_seed(args.seed)

if len(args.gpu_ids) > 0:
    torch.cuda.set_device(args.gpu_ids[0])
    cudnn.benchmark = True
    print(args)

# visualizer = Visualizer()
train_dataset = dataset.DatasetTrain(args)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)
test_dataset = dataset.DatasetTest(args)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

if args.model == 'srfusemodel':
    model = SRFuseModel()

else:
    raise TypeError('{} model is not available!'.format(args.model))
model.initialize(args)    # model parameter setting e.g. load pre-trained model
model.load_model()
best = 0
timer_data, timer_model = timer(), timer()

for epoch in range(model.epoch+1, args.nEpochs+1):
    model.set_mode(train=True)
    model.set_epoch(epoch)
    for i, data in enumerate(train_loader, 1):
        # break
        model.set_input(data)
        timer_data.hold()
        timer_model.tic()
        loss = model.train()
        timer_model.hold()

        if i % args.print_every == 0:
            loss_log = 'epoch: {}, iteration: {}/{} \t{:.1f}+{:.1f}s,\t'.format(epoch, i, len(train_loader), timer_data.release(), timer_model.release())
            for loss_type, loss_v in loss.items():
                loss_log += '{} : {}'.format(loss_type, loss_v)

            print(loss_log)
            model.loss_record(loss)
            # visualizer.plot_loss(model.losses)
        if i % args.save_every == 0:
            images = model.get_results()
            # visualizer.display_results(images)


    model.scheduler1.step()  # update learning rate
    # model.scheduler2.step()  # update learning rate
    # print('Learning rate: %f' % model.scheduler1.get_last_lr()[0])
    print('ok')
    # print('Learning rate: %f' % model.scheduler2.get_last_lr()[0])

    if epoch % args.save_epoch == 0:
        print('evaluation')
        model.set_mode(train=False)
        model.set_epoch(epoch)
        # average_psnr = []
        # average_ssim = []
        flag = 0
        num = 0
        psnr_list=[]
        ssim_list=[]
        psnr_ave_list=[]
        ssim_ave_list=[]
        for i, data in enumerate(test_loader):
            model.set_eval_input(data)
            outputs = model.eval()  # torch.Size([3, 720, 1280])

            # model.save_image(outputs,'/home/vista/luZiTao/python_code/compettion/NTIRE2021/mid_out/005.png')
            # print(outputs['output'].size(), outputs['target'].size())
            # psnr_, ssim_ = model.comput_PSNR_SSIM(outputs['output'], outputs['target'])
            gt, pred = tensor2Np([outputs['target'],outputs['output']], 255.)
            psnr1 = peak_signal_noise_ratio(pred,gt)
            ssim1 = structural_similarity(pred,gt, multichannel=True, gaussian_weights=True, use_sample_covariance=False)
            psnr_list.append(psnr1)
            ssim_list.append(ssim1)
            flag += 1
            if flag % 100 == 0:
                psnr_1 = psnr_list[num*100:num*100+100]
                psnr_ave = np.average(psnr_1)
                psnr_ave_list.append(psnr_ave)
                ssim_1 = ssim_list[num*100:num*100+100]
                num += 1
                ssim_ave = np.average(ssim_1)
                ssim_ave_list.append(ssim_ave)
                log = 'Epoch %d:  Average psnr: %f  ssim: %f \n' % (epoch, psnr_ave, ssim_ave)
                print(log)
                f = open('/raid/LZT/python_code/my_1/checkpoints_1/srfusemodel/log_SR_4.txt','a')
                f.write(log)
                f.close()
        average_psnr = np.average(psnr_ave_list)
        average_ssim = np.average(ssim_ave_list)
        log = 'Epoch %d: All Average psnr: %f , ssim: %f \n' % (epoch, average_psnr, average_ssim)
        print(log)
        model.log_file.write(log)
        f = open('/raid/LZT/python_code/my_1/checkpoints_1/srfusemodel/log_SR_4.txt','a')
        f.write('--------------------------------------------------------------')
        f.write(log)
        f.close()


        if average_psnr > best:
            is_best = True
            best = average_psnr
        else:
            is_best = False

        model.save_model(is_best)

model.log_file.close()
