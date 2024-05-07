import torch
import os
import torch.nn as nn
from utility import mkdir
from data.common import tensor2Np
import numpy as np
import skimage.color as sc
import cv2
import math
import imageio
from importlib import import_module

class BaseModel():
    def initialize(self, args):
        print('Making model....')

        self.args = args
        self.gpu_ids = args.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.task = args.task
        self.model_name = args.model
        self.test_final = args.test_final
        self.scale_param = args.scale_param
        self.stage = args.stage
        self.train_phase = args.train_phase

        # save model path
        self.save_dir = os.path.join('checkpoints_1', self.model_name)
        mkdir(self.save_dir)

        # save results path
        self.result_dir = os.path.join(self.save_dir, 'results')
        mkdir(self.result_dir)

        save_name = '/log_{}_{}.txt'.format(self.task, self.scale_param)
        open_type = 'a' if os.path.exists(self.save_dir + save_name) else 'w'
        self.log_file = open(self.save_dir + save_name, open_type)
        self.log_file.write('\n')
        for arg in vars(args):
            self.log_file.write('{}:{}\n'.format(arg, getattr(args, arg)))
        self.log_file.write('\n')

        if self.stage == 'train':
            self.losses = {}  # initialize loss record
            for loss_type in self.args.loss.split('+'):
                self.losses.update({loss_type: []})
    def load_model(self):
        if self.stage == 'train':
            if self.args.pre_train != '.':
                self.load_model_(self.args.pre_train)
                self.args.resume = 0
                self.set_epoch(self.args.resume)
            elif self.args.resume > 0:
                self.load_model_('epoch{}'.format(self.args.resume))
                self.set_epoch(self.args.resume)
            else:
                self.set_epoch(0)
        elif self.stage == 'test':
            self.load_model_(self.args.test_final)

        else:
            if self.args.resume > 0:
                self.load_model_('epoch{}'.format(self.args.resume))
            else:
                self.load_model_('best')


    def get_results(self):
        images = {'input': self.input, 'output': self.output,
                  'target': self.target}

        return images

    def loss_record(self, losses):

        for loss_type, loss in losses.items():
            self.losses[loss_type].append(loss)

    def save_model(self, is_best=False):
        save_name = '{}_{}_{}'.format(self.model_name, self.task, self.scale_param)
        net = self.model
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():

            if is_best:
                save_name_ = save_name + 'best.pt'
                torch.save(
                    net.module.cpu().state_dict(),
                    os.path.join(self.save_dir, save_name_),
                    _use_new_zipfile_serialization=False
                )
                print(save_name)

            save_name_ = save_name + 'epoch{}.pt'.format(self.epoch)
            torch.save(
                net.module.cpu().state_dict(),
                os.path.join(self.save_dir, save_name_),
                _use_new_zipfile_serialization=False
            )

            print(save_name)
            net.cuda(self.gpu_ids[0])

        else:
            torch.save(net.cpu().state_dict(), os.path.join(self.save_dir, save_name),_use_new_zipfile_serialization=False)
            if is_best:
                save_name_ = save_name + 'best.pt'
                torch.save(net.cpu().state_dict(), os.path.join(self.save_dir, save_name_),_use_new_zipfile_serialization=False)

            save_name_ = save_name + 'epoch{}.pt'.format(self.epoch)
            torch.save(net.cpu().state_dict(), os.path.join(self.save_dir, save_name_),_use_new_zipfile_serialization=False)
            print(save_name)


    def load_model_(self, load_n):
        net = self.model
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        load_name = '{}_{}_{}_{}.pt'.format(self.model_name, self.task, self.scale_param, load_n)
        load_path = os.path.join(self.save_dir, load_name)
        print('loading the model from %s' % load_path)

        state_dict = torch.load(load_path, map_location=str(self.device))
        net.load_state_dict(state_dict, strict=False)   # ignore unexpected_kyes or missing_keys


    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_mode(self, train):
        if train:
            self.model.train()
        else:
            self.model.eval()

    def set_input(self, input):
        self.input = input['input'].to(self.device)
        self.target = input['target'].to(self.device)
        self.filename = input['filename']

    def set_eval_input(self, input):
        self.eval_input = input['input'].to(self.device)
        self.eval_target = input['target'].to(self.device)
        self.eval_filename = input['filename']
    def set_test_input(self, input):
        self.test_input = input['input'].to(self.device)
        self.test_filename = input['filename']
    def set_val_psnr_input(self,input):
        self.test_input = input

    def train(self):
        pass

    def eval(self):
        pass



    def loss_define(self):
        loss = {}
        for loss_type in self.args.loss.split('+'):
            if loss_type == 'MSE':
                loss_function = nn.MSELoss(reduction='mean')
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'Cross':
                loss_function = nn.CrossEntropyLoss()
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=self.args.rgb_rage
                )
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    self.args,
                    loss_type
                )
            else:
                raise TypeError('{} loss is not callable!'.format(loss_type))

            loss.update({loss_type: loss_function})
        return loss


    def save_image(self, img_tensor, save_name):
        img_np = tensor2Np([img_tensor], self.args.rgb_range)
        imageio.imsave(save_name, img_np[0])
        print('Saving image into {}'.format(save_name))



    def comput_PSNR_SSIM(self, pred, gt, shave_border=0):
        # print(pred)
        # print(gt)
        pred, gt = tensor2Np([pred, gt], self.args.rgb_range)

        pred = pred.astype(np.float32)
        gt = gt.astype(np.float32)

        height, width = pred.shape[:2]
        pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]  #720 1280 3
        gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
        gt = gt / 255.
        pred = pred / 255.
        if pred.shape[2] == 3 and gt.shape[2] == 3:
            pred_y = sc.rgb2ycbcr(pred)[:, :, 0]   ##########要输入0-1
            gt_y = sc.rgb2ycbcr(gt)[:, :, 0]
        elif pred.shape[2] == 1 and gt.shape[2] == 1:
            pred_y = pred[:, :, 0]
            gt_y = gt[:, :, 0]
        else:
            raise ValueError('Input or output channel is not 1 or 3!')

        psnr_ = self._calc_PSNR(pred_y, gt_y)
        ssim_ = self._calc_ssim(pred_y, gt_y)

        return psnr_, ssim_

    def _ssim(self, img1, img2):

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    def _calc_ssim(self, img1, img2):
        """
        calculate SSIM the same as matlab
        input [0, 255]

        :return:
        """
        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return self._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(self._ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self._ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimension')

    def _calc_PSNR(self, pred, gt):
        """
        calculate PSNR the same as matlab
        input [0, 255] float
        :param pred:
        :param gt:
        :return:
        """
        if not pred.shape == gt.shape:
            raise ValueError('Input images must have the same dimensions.')
        if pred.ndim != 2:
            raise ValueError('Input images must be H*W.')


        imdff = pred - gt
        # print(imdff)
        rmse = math.sqrt(np.mean(imdff ** 2))
        # print(rmse)
        if rmse == 0:
            return 100
        return 20 * math.log10(255.0 / rmse)