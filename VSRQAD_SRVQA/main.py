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

torch.manual_seed(args.seed)

if len(args.gpu_ids) > 0:
    torch.cuda.set_device(args.gpu_ids[0])
    cudnn.benchmark = True
    print(args)

train_dataset = dataset.DatasetTrain(args)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
test_dataset = dataset.DatasetTest(args)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=16)

if args.model == 'dreamSRVQA':
    model = dream_SRVQA()
else:
    raise TypeError('{} model is not available!'.format(args.model))
model.initialize(args)    # model parameter setting e.g. load pre-trained model
model.load_model()
max_srcc = 0
best = 100000
timer_data, timer_model = timer(), timer()

for epoch in range(model.epoch+1, args.nEpochs+1):
    model.set_mode(train=True)
    model.set_epoch(epoch)
    gloss = 0
    for i, data in enumerate(train_loader, 1):
       # break
        model.set_input(data)
        timer_data.hold()
        timer_model.tic()
        loss = model.train()
        gloss += loss['LOSS']
        timer_model.hold()
        if i % args.print_every == 0:
            loss_log = 'epoch: {}, iteration: {}/{} \t{:.1f}+{:.1f}s,\t'.format(epoch, i, len(train_loader), timer_data.release(), timer_model.release())
            for loss_type, loss_v in loss.items():
                loss_log += '{} : {}'.format(loss_type, loss_v)
            print(loss_log)
    print('TRAIN: epoch [{}] Train_Loss [{:.8f}]'.format(epoch, gloss / i))

    model.scheduler1.step()  # update learning rate
    # model.scheduler2.step()  # update learning rate
    # print('ok')
    if epoch % args.save_epoch == 0:
        print('evaluation')
        model.set_mode(train=False)
        model.set_epoch(epoch)
        flag = 0
        num = 0
        # val_loss = 0
        ave_loss = []
        MOS = []
        OS = []
        is_best = False
        for i, data in enumerate(test_loader):
            # break        self.model_fea = nn.Sequential(*list(self.pre_resnet.children())[:-1])
            model.set_eval_input(data)
            pre_mos, mos, loss = model.eval()  # torch.Size([3, 720, 1280])\
            for k in range(len(pre_mos)):
                MOS.append(mos[k].item())
                OS.append(pre_mos[k].item())
            # MOS.append(mos)
            # OS.append(pre_mos)
            ave_loss.append(loss)

        srcc = stats.spearmanr(np.array(MOS), np.array(OS))
        plcc = stats.pearsonr(np.array(MOS), np.array(OS))
        rmse_sum = 0
        for w in range(len(MOS)):
            rmse_sum += (MOS[w] - OS[w]) * (MOS[w] - OS[w])
        rmse = np.sqrt(rmse_sum / len(MOS))
        if max_srcc < srcc.correlation:
            max_srcc = srcc.correlation
            is_best = True

        log = 'TEST: epoch [{}] SRCC [{:.4f}] PLCC [{:.4f}] Max_SRCC [{:.4f}] val_loss [{:.4f}]'.format(epoch,srcc.correlation,plcc[0],max_srcc,rmse)
        print(log)
        f = open('./checkpoints_1/dreamSRVQA/log_SR_4.txt', 'a')
        f.write(log)
        f.close()

        model.save_model(is_best)

model.log_file.close()
