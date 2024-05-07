
import math
from model import common
from model.basemodel import BaseModel
import torch.nn as nn
import torch
from torch.nn import functional as F
import itertools
from data.common import quantize
from utility import make_optimizer1, make_scheduler1,MeanShift,Upsampler,make_optimizer2,make_scheduler2
import os
import model
import datetime
import time
from data.common import save_image
class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)
        self.frames_num = args.frames_num

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body1 = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblock)
        ]

        m_body1.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, args.scale_param, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body1)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, input):
        inputs = torch.chunk(input, self.frames_num, dim=1)
        outputs = []
        for input_ in inputs:

            x = self.head(input_)
            res = self.body(x)
            res += x
            y = self.tail(res)
            outputs.append(y)

        out = torch.cat(outputs, dim=1)
        return out



class PreFuse(nn.Module):
    def __init__(self,args,conv=common.default_conv):
        super(PreFuse, self).__init__()
        n_feats = args.n_feats//2
        self.conv2 = conv(n_feats,n_feats,kernel_size=5)
        self.conv3 = conv(n_feats,n_feats,kernel_size=3)
        self.conv4 = conv(n_feats,n_feats,kernel_size=1)
        self.ESA = ESA(n_feats)
    def forward(self, input):
        # out1 = self.conv1(input)
        input2 = input
        for i in range(3):
            out2 = self.conv2(input2)
            out3 = self.conv3(out2)
            out4 = self.conv4(out3)
            input2 = self.ESA(out4)
        out = input + input2
        return out



class FuseModel(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(FuseModel, self).__init__()

        n_feats = args.n_feats
        act = nn.ReLU(True)
        self.frames_num = args.frames_num

        self.nonlocal_warp = NonLocalModule(kernel_size=3, n_colors=args.n_colors)

        self.conv0 = conv(args.n_colors,n_feats,kernel_size=7)
        self.conv1 = conv(n_feats, n_feats, kernel_size=3)
        self.conv2 = conv(n_feats*5, n_feats, kernel_size=1)
        self.conv3 = conv(n_feats, n_feats, kernel_size=3)
        self.conv4 = conv(n_feats, args.n_colors, kernel_size=1)
        self.conv5 = conv(n_feats*2,n_feats,kernel_size=1)
        self.conv6 = conv(n_feats,n_feats,kernel_size=1)
        self.conv7 = conv(args.n_colors,n_feats,kernel_size=3)
        self.conv8 = conv(n_feats * 5, n_feats, kernel_size=3)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.PreFuse = PreFuse(args)
        self.ESA = ESA(n_feats)
    # def forward(self, input):       #通道32
    #     inputs = torch.chunk(input, self.frames_num, dim=1) #tuple 5
    #
    #     k = len(inputs)//2
    #     output_list = []
    #     output_list2 = []
    #     for i in range(len(inputs)):
    #         if i == k:
    #             output_ = inputs[k]
    #         else:
    #             output_ = self.nonlocal_warp(inputs[i], inputs[k])      #torch 1,3,192,192
    #         output_ = self.conv0(output_)
    #         output_list2.append(output_)
    #
    #
    #         # output_2 = self.head(output_)
    #         # output_list.append(output_2)    #1 32 192 192
    #     out_frame1 = self.PreFuse(output_list2[0])
    #     out_frame2 = self.PreFuse(output_list2[1])
    #     out_frame3 = output_list2[2]
    #     out_frame4 = self.PreFuse(output_list2[3])
    #     out_frame5 = self.PreFuse(output_list2[4])
    #     map1 = abs(output_list2[0]-output_list2[2])/255
    #     map2 = abs(output_list2[1]-output_list2[2])/255
    #     map4 = abs(output_list2[3]-output_list2[2])/255
    #     map5 = abs(output_list2[4]-output_list2[2])/255
    #     final_frame1 = out_frame1 * map1 + out_frame3 * (1-map1)
    #     final_frame2 = out_frame2 * map2 + out_frame3 * (1-map2)
    #     final_frame4 = out_frame4 * map4 + out_frame3 * (1-map4)
    #     final_frame5 = out_frame5 * map5 + out_frame3 * (1-map5)
    #     output_list = [final_frame1,final_frame2,out_frame3,final_frame4,final_frame5]
    #
    #     # first_0 = self.conv5(output_list[0])
    #     # first_1 = self.conv5(output_list[1])
    #     # first_2 = self.conv5(output_list[2])
    #     # first_3 = self.conv5(output_list[3])
    #     # first_4 = self.conv5(output_list[4])
    #
    #
    #     for j in range(5):
    #         input_0 = output_list[0]
    #         input_1 = output_list[1]
    #         input_2 = output_list[2]
    #         input_3 = output_list[3]
    #         input_4 = output_list[4]
    #         mid_0 = self.conv1(input_0)
    #         mid_1 = self.conv1(input_1)
    #         mid_2 = self.conv1(input_2)
    #         mid_3 = self.conv1(input_3)
    #         mid_4 = self.conv1(input_4)
    #         mid_list = [mid_0,mid_1,mid_2,mid_3,mid_4]
    #         mid_cat = torch.cat(mid_list,dim=1)
    #         mid_cat2 = self.conv2(mid_cat)
    #         mid2_0 = torch.cat((mid_0, mid_cat2),dim=1)
    #         mid2_1 = torch.cat((mid_1, mid_cat2),dim=1)
    #         mid2_2 = torch.cat((mid_2, mid_cat2),dim=1)
    #         mid2_3 = torch.cat((mid_3, mid_cat2),dim=1)
    #         mid2_4 = torch.cat((mid_4, mid_cat2),dim=1)
    #         out_0 = self.conv3(mid2_0) + input_0
    #         out_1 = self.conv3(mid2_1) + input_1
    #         out_2 = self.conv3(mid2_2) + input_2
    #         out_3 = self.conv3(mid2_3) + input_3
    #         out_4 = self.conv3(mid2_4) + input_4
    #
    #         output_list = [out_0,out_1,out_2,out_3,out_4]
    #     out_mid=self.conv4(output_list[2])
    #     return out_mid

    def forward(self, input):     #简单版本
        inputs = torch.chunk(input, self.frames_num, dim=1) #tuple 5

        k = len(inputs)//2
        output_list = []
        output_list2 = []
        # for i in range(len(inputs)):  //原来的
        #     if i == k:
        #         output_ = inputs[k]
        #     else:
        #         output_ = self.nonlocal_warp(inputs[i], inputs[k])      #torch 1,3,192,192
        #         warp_img = quantize(output_, 255)
        #         save_image(warp_img, 'E:\\luzitao\\test_red_noise\SR\\{}.png'.format(i))
        #     output_list2.append(output_)
        for i in range(len(inputs)):    #warp中间帧
            output_ = self.nonlocal_warp(inputs[i], inputs[k])      #torch 1,3,192,192
            warp_img = quantize(output_, 255)
            save_image(warp_img, 'E:\\luzitao\\test_red_noise\SR\\{}.png'.format(i))
            output_list2.append(output_)

        out_frame3 = output_list2[2]

        map1 = abs(output_list2[0]-output_list2[2])/abs(torch.max(output_list2[0]-output_list2[2]))
        map1 = torch.mean(map1,dim=1,keepdim=True)
        map2 = abs(output_list2[1]-output_list2[2])/abs(torch.max(output_list2[1]-output_list2[2]))
        map2 = torch.mean(map2,dim=1,keepdim=True)
        map4 = abs(output_list2[3]-output_list2[2])/abs(torch.max(output_list2[3]-output_list2[2]))
        map4 = torch.mean(map4,dim=1,keepdim=True)
        map5 = abs(output_list2[4]-output_list2[2])/abs(torch.max(output_list2[4]-output_list2[2]))
        map5 = torch.mean(map5,dim=1,keepdim=True)

        final_frame1 = output_list2[2] * map1 + output_list2[0] * (1-map1)
        warp_img = quantize(final_frame1,255)
        save_image(warp_img,'E:\\luzitao\\test_red_noise\SR\\warp.png')
        final_frame2 = output_list2[2] * map2 + output_list2[1] * (1-map2)
        final_frame4 = output_list2[2] * map4 + output_list2[3] * (1-map4)
        final_frame5 = output_list2[2] * map5 + output_list2[4] * (1-map5)
        final_frame_list = [final_frame1,final_frame2,out_frame3,final_frame4,final_frame5]
        for i in range(5):
            mid_frame = self.conv0(final_frame_list[i])
            output_list.append(mid_frame)
        output_list_cat = torch.cat(output_list,dim=1)
        pre_input = self.lrelu(self.conv8(output_list_cat))  # f

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #
        ave_img = (final_frame1 + final_frame2 + out_frame3 + final_frame4 + final_frame5) / 5
        ave_img = self.lrelu(self.conv7(ave_img))
        ave_img_ori = ave_img
        for j in range(3):
            input_img = ave_img
            mid_img = self.conv1(input_img)
            mid_img = torch.cat((mid_img,pre_input),dim=1)
            mid_img = self.conv5(mid_img)
            mid_img = self.ESA(mid_img)
            mid_img = self.relu(self.conv3(mid_img)) + ave_img_ori
            ave_img = mid_img
        out_mid = self.conv4(ave_img)
        #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # #---------------------------------------------
        # att_max = self.maxpool(pre_input)
        # att_avg = self.avgpool(pre_input)
        # att = self.lrelu(self.conv5(torch.cat((att_avg, att_max), dim=1)))
        # att = self.lrelu(self.conv6(att))
        # att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        # ave_img = (final_frame1 + final_frame2 + out_frame3 + final_frame4 + final_frame5) / 5
        # # ave_img_out = quantize(ave_img, 255)
        # # save_image(ave_img_out, 'E:\\luzitao\\test_red_noise\\SR\\ave_img.png')
        # ave_img = self.conv7(ave_img)
        # ave_img = self.lrelu(self.conv1(ave_img))
        # ave_img_ori = ave_img
        # for j in range(3):
        #     input_img = ave_img
        #     mid_img = self.conv1(input_img)
        #     mid_img = torch.cat((mid_img,att),dim=1)
        #     mid_img = self.conv5(mid_img)
        #     mid_img = self.ESA(mid_img)
        #     mid_img = self.relu(self.conv3(mid_img)) + ave_img_ori
        #     ave_img = mid_img
        # out_mid=self.conv4(ave_img)
        #-------------------------------------------------------------
        return out_mid

class NonLocalModule(nn.Module):
    def __init__(self, kernel_size, n_colors):
        super(NonLocalModule, self).__init__()
        self.kernel_size = kernel_size
        self.n_colors = n_colors
        kernel_s = kernel_size * kernel_size * n_colors
        kernel = []
        for i in range(kernel_s):
            k = torch.LongTensor([i]).reshape(1, 1)
            one_hot = torch.zeros(1, kernel_s).scatter_(1, k, 1).reshape(1, n_colors, kernel_size, kernel_size)
            kernel.append(one_hot)
        kernel = torch.cat(kernel, dim=0)
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.weights = None


    def forward_chop(self, input_img, refer_img):         # input: frame 1, 2, 4, 5   refer: frame 3
        input_patch = F.conv2d(input_img, self.kernel, stride=1, padding=0)   #? self.kernel [b,27,3,3]
        refer_patch = F.conv2d(refer_img, self.kernel, stride=1, padding=0)

        h = 0.5
        input_patch_reshape = input_patch.view(input_patch.size(0), input_patch.size(1), -1)      # B x C*k*k x H*W [b,27,2116]
        input_patch_reshape_T = input_patch_reshape.permute(0, 2, 1)    # B x H*W x C*k*k
        refer_patch_reshape_T = refer_patch.view(refer_patch.size(0), input_patch.size(1), -1).permute(0, 2, 1)   # B x H*W x C*k*k

        XY = torch.matmul(refer_patch_reshape_T, input_patch_reshape)
        XX = torch.sum(refer_patch_reshape_T * refer_patch_reshape_T, dim=2, keepdim=True)
        YY = torch.sum(input_patch_reshape * input_patch_reshape, dim=1, keepdim=True)
        attention = (2 * XY - XX - YY).div(h * h + 0.1)
        attention = F.softmax(attention, dim=-1)# B H*W H*W

        output = torch.matmul(attention, input_patch_reshape_T) #B H*W C*k*k
        output = output.permute(0, 2, 1).contiguous() #B C*k*k H*W
        output = output.view_as(input_patch)  #?B C*k*k H W

        output, self.weights = self.img_sum(output, input_img, self.kernel_size, self.weights)

        return output

    def forward(self, x, y, shave=15, min_size=20000):
        scale = 1
        b, c, h, w = x.size()

        if h*w < min_size:
            output = self.forward_chop(x, y)
            return output

        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        input_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        refer_list = [
            y[:, :, 0:h_size, 0:w_size],
            y[:, :, 0:h_size, (w - w_size):w],
            y[:, :, (h - h_size):h, 0:w_size],
            y[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            output_list = []
            for i in range(0, 4, 1):
                input_batch = torch.cat(input_list[i:(i+1)], dim=0)
                refer_batch = torch.cat(refer_list[i:(i+1)], dim=0)
                out_batch = self.forward_chop(input_batch, refer_batch)
                output_list.extend(out_batch.chunk(1, dim=0))
        else:
            output_list = [
                self.forward(input_list[i], refer_list[i], shave=shave, min_size=min_size) \
                for i in range(len(input_list))
            ]


        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = output_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = output_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = output_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = output_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def img_sum(self, input_patch, input_img, kernel_size, weights_out):

        output = torch.zeros(input_img.size()).to(input_img.device)

        if (weights_out is None) or (weights_out.size(2) != input_img.size(2)) or (
                weights_out.size(3) != input_img.size(3)):
            weights = torch.ones(1, kernel_size * kernel_size, input_patch.size(2), input_patch.size(3))
            weights_output = torch.zeros(1, 1, input_img.size(2), input_img.size(3))
        else:
            weights_output = weights_out

        for i in range(kernel_size):
            for j in range(kernel_size):
                in_x = input_patch[:, :, i::kernel_size, j::kernel_size]
                in_x = F.pixel_shuffle(in_x, kernel_size)
                output[:, :, i:i + in_x.size(2), j:j + in_x.size(3)] += in_x

                if (weights_out is None) or (weights_out.size(2) != input_img.size(2)) or (
                        weights_out.size(3) != input_img.size(3)):
                    wei_x = weights[:, :, i::kernel_size, j::kernel_size]
                    wei_x = F.pixel_shuffle(wei_x, kernel_size)
                    weights_output[:, :, i:i + wei_x.size(2), j:j + wei_x.size(3)] += wei_x

        weights_output = weights_output.to(output.device)
        output = output / weights_output

        return output, weights_output

class SpatialNet(nn.Module):
    def __init__(self, args):
        super(SpatialNet, self).__init__()
        print('ok')
        n_feats = args.n_feats
        kernel_sizes = 3
        num_classes = 3
        act = nn.ReLU(True)
        conv = common.default_conv
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = conv(args.n_colors, n_feats, 5)
        self.conv2 = conv(n_feats, 1, 3)
        self.resnet1 = common.ResBlock(conv, n_feat=n_feats, kernel_size=kernel_sizes, act=act, res_scale=args.res_scale)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, input):  # 训练的时候可能要先训练前面的feature部分，后面的倍数部分需要冻结前面feature的参数
        x = self.conv1(input)
        x = self.resnet1(x)
        fea = self.conv2(x)
        out = fea + input     # 这里加上的是单通道的，考虑把输入变成单通道，或者将fea复制3次,可能还要进行pooling
        return out

class SRFuseModel(BaseModel):
    def initialize(self, args):
        BaseModel.initialize(self, args)

        self.spatial_model = SpatialNet(args)
        self.spatial_model = nn.DataParallel(self.spatial_model, device_ids=args.gpu_ids)
        self.model_sr = RFANet(args)
        self.model_sr = nn.DataParallel(self.model_sr, device_ids=args.gpu_ids)
        self.model_fuse = FuseModel(args)
        self.model_fuse = nn.DataParallel(self.model_fuse, device_ids=args.gpu_ids)
        self.model_sr.to(self.device)
        self.criterion = self.loss_define()
        self.cur_frame = args.frames_num // 2
        self.n_colors = args.n_colors
        if self.train_phase == 'sr':
            self.optimizer1 = make_optimizer1(args, self.model_sr.parameters())
            self.scheduler1 = make_scheduler1(args,self.optimizer1)
        elif self.train_phase == 'fuse':
            self.optimizer2 = make_optimizer2(args, self.model_fuse.parameters())
            self.scheduler2 = make_scheduler2(args,self.optimizer2)
        else:
            self.optimizer1 = make_optimizer1(args, self.model_sr.parameters())
            # self.optimizer = make_optimizer(args, itertools.chain(self.model_sr.parameters(), self.model_fuse.parameters()))
            self.optimizer2 = make_optimizer2(args, self.model_fuse.parameters())
            self.scheduler1 = make_scheduler1(args,self.optimizer1)
            self.scheduler2 = make_scheduler2(args,self.optimizer2)
            print('ok')
    # def eval_initialize(self):

    def train(self):
        if self.train_phase == 'sr':
            sr_out = self.model_sr(self.input)
            loss = 0
            # loss = self.criterion(sr_out, self.target)
            for (loss_type, loss_fun) in self.criterion.items():
                loss1 = loss_fun(sr_out, self.target[:, self.cur_frame * self.n_colors:(self.cur_frame+1) * self.n_colors, :, :])
                loss = loss1 + loss
                losses = {loss_type: loss1.item()}
                # print(loss_type)

            self.output = quantize(sr_out, self.args.rgb_range)
            self.optimizer1.zero_grad()
            loss.backward()
            self.optimizer1.step()

        elif self.train_phase == 'fuse':
            with torch.no_grad():
                sr_out = self.model_sr(self.input)

            # sr_out[:,0:3,:,:] = abs()
            fuse_out = self.model_fuse(sr_out)
            loss = 0
            for (loss_type, loss_fun) in self.criterion.items():
                # loss2 = loss_fun(sr_out,self.target)
                loss1 = loss_fun(fuse_out, self.target[:, self.cur_frame * self.n_colors:(self.cur_frame+1) * self.n_colors, :, :])
                loss = loss1 + loss
                losses = {loss_type: loss1.item()}
                # loss = self.criterion(fuse_out, self.target[:, self.cur_frame * self.args.n_colors:(self.cur_frame+1) * self.args.n_colors, :, :])
            self.output = quantize(fuse_out, self.args.rgb_range)
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer2.step()

        else:
            sr_out = self.model_sr(self.input)
            fuse_out = self.model_fuse(sr_out)
            loss = 0
            for (loss_type, loss_fun) in self.criterion.items():
                loss_1 = loss_fun(sr_out, self.target)
                loss_2 = loss_fun(fuse_out, self.target[:, self.cur_frame * self.n_colors:(self.cur_frame+1) * self.n_colors, :, :])
                loss1 = loss_1*0.5 + loss_2
                loss = loss1 + loss
                losses = {loss_type: loss1.item()}

            self.output = quantize(fuse_out, self.args.rgb_range)

        # self.optimizer.zero_grad()
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()

        # self.optimizer.step()

        return losses

    def eval(self):
        with torch.no_grad():
            if self.train_phase == 'sr':
                sr_out = self.model_sr(self.eval_input)     #1 15 720 1280
                output = sr_out   #1 3 720 1280
            elif self.train_phase == 'fuse':
                sr_out = self.model_sr(self.eval_input)
                # sr_out_mid = sr_out[:, self.cur_frame * self.n_colors:(self.cur_frame+1) * self.n_colors, :, :]
                # save_image(sr_out_mid[0],'/home/vista/luZiTao/python_code/compettion/NTIRE2021/mid_out/003.png')
                fuse_out = self.model_fuse(sr_out)
                # fuse_out = quantize(fuse_out, self.args.rgb_range)
                # save_image(fuse_out[0],'/home/vista/luZiTao/python_code/compettion/NTIRE2021/mid_out/004.png')
                output = fuse_out
            else:
                sr_out = self.model_sr(self.eval_input)
                fuse_out = self.model_fuse(sr_out)
                output = fuse_out
        output = quantize(output, self.args.rgb_range)  #clamp
        # return{'output': output[0]}
        return {'input': self.eval_input[0][self.cur_frame * self.n_colors:(self.cur_frame+1) * self.n_colors, :, :],
                'output': output[0], 'target': self.eval_target[0][self.cur_frame * self.n_colors:(self.cur_frame+1) * self.n_colors, :, :]}

    def test(self):
        with torch.no_grad():
            if self.train_phase == 'sr':

                # start = time.clock()
                sr_out = self.model_sr(self.eval_input)
                # end = time.clock()
                # print(end-start)
                output = sr_out
            elif self.train_phase == 'fuse':
                sr_out = self.model_sr(self.test_input)

                # sr_out_mid = sr_out[:, self.cur_frame * self.n_colors:(self.cur_frame+1) * self.n_colors, :, :]
                # save_image(sr_out_mid[0],'/home/vista/luZiTao/python_code/compettion/NTIRE2021/mid_out/003.png')
                # print(sr_out_mid[0])
                fuse_out = self.model_fuse(sr_out)
                # fuse_out = self.model_fuse(self.test_input)
                # save_image(fuse_out[0],'/home/vista/luZiTao/python_code/compettion/NTIRE2021/mid_out/004.png')
                output = fuse_out
            else:
                sr_out = self.model_sr(self.test_input)
                fuse_out = self.model_fuse(sr_out)
                output = fuse_out
        output = quantize(output, self.args.rgb_range)  #clamp

        # return {'output': output[0]}

        return {'output': output[0], 'target':self.eval_target[0][self.cur_frame * self.n_colors:(self.cur_frame+1) * self.n_colors, :, :]}

    def save_model(self, is_best=False):
        save_name = '{}_{}_{}'.format(self.model_name, self.task, self.scale_param)
        if self.train_phase == 'sr':
            net = self.model_sr
            save_name_ = save_name + '_sr_epoch{}.pt'.format(self.epoch)
            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(
                    {'model_sr_state_dict': net.module.cpu().state_dict()},
                    os.path.join(self.save_dir, save_name_),
                    _use_new_zipfile_serialization=False
                )
                if is_best:
                    save_name_ = save_name + '_sr_best.pt'
                    torch.save(
                        {'model_sr_state_dict': net.module.cpu().state_dict()},
                        os.path.join(self.save_dir, save_name_),
                        _use_new_zipfile_serialization=False
                    )
                    print(save_name)

                print(save_name)
                net.cuda(self.gpu_ids[0])

            else:
                if is_best:
                    save_name_ = save_name + '_sr_best.pt'
                    torch.save({'model_sr_state_dict': net.cpu().state_dict()}, os.path.join(self.save_dir, save_name_),_use_new_zipfile_serialization=False)
                    print(save_name)

                torch.save({'model_sr_state_dict':net.cpu().state_dict()}, os.path.join(self.save_dir, save_name),_use_new_zipfile_serialization=False)
                torch.save(net.cpu().state_dict(), os.path.join(self.save_dir, save_name_),_use_new_zipfile_serialization=False)
                print(save_name)

        elif self.train_phase == 'fuse':
            net1 = self.model_sr
            net2 = self.model_fuse
            save_name_ = save_name + '_fuse_epoch{}.pt'.format(self.epoch)

            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(
                    {'model_sr_state_dict': net1.module.cpu().state_dict(),
                     'model_fuse_state_dict': net2.module.cpu().state_dict()},
                    os.path.join(self.save_dir, save_name_)
                )
                if is_best:
                    save_name_ = save_name + '_fuse_best.pt'
                    torch.save(
                        {'model_sr_state_dict': net1.module.cpu().state_dict(),
                         'model_fuse_state_dict': net2.module.cpu().state_dict()},
                        os.path.join(self.save_dir, save_name_)
                    )
                    print(save_name)



                print(save_name)
                net1.cuda(self.gpu_ids[0])
                net2.cuda(self.gpu_ids[0])

            else:
                if is_best:
                    save_name_ = save_name + '_fuse_best.pt'
                    torch.save({'model_sr_state_dict': net1.cpu().state_dict(),
                                'model_fuse_state_dict': net2.cpu().state_dict(),
                                }, os.path.join(self.save_dir, save_name_))
                    print(save_name)

                torch.save({'model_sr_state_dict': net1.cpu().state_dict(),
                                'model_fuse_state_dict': net2.cpu().state_dict(),
                                }, os.path.join(self.save_dir, save_name))
                # torch.save(net.cpu().state_dict(), os.path.join(self.save_dir, save_name_))
                print(save_name)
        else:
            net1 = self.model_sr
            net2 = self.model_fuse
            save_name_ = save_name + '_joint_epoch{}.pt'.format(self.epoch)

            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(
                    {'model_sr_state_dict':net1.module.cpu().state_dict(),
                     'model_fuse_state_dict':net2.module.cpu().state_dict()},
                    os.path.join(self.save_dir, save_name_)
                )
                if is_best:
                    save_name_ = save_name + '_joint_best.pt'
                    torch.save(
                        {'model_sr_state_dict': net1.module.cpu().state_dict(),
                         'model_fuse_state_dict': net2.module.cpu().state_dict()},
                        os.path.join(self.save_dir, save_name_)
                    )
                    print(save_name)


                print(save_name)
                net1.cuda(self.gpu_ids[0])
                net2.cuda(self.gpu_ids[0])

            else:
                if is_best:
                    save_name_ = save_name + '_joint_best.pt'
                    torch.save({'model_sr_state_dict':net1.cpu().state_dict(),
                                'model_fuse_state_dict':net2.cpu().state_dict()},
                               os.path.join(self.save_dir, save_name_))
                    print(save_name)

                torch.save(
                    {'model_sr_state_dict': net1.cpu().state_dict(),
                     'model_fuse_state_dict': net2.cpu().state_dict()},
                    os.path.join(self.save_dir, save_name_)
                )
                print(save_name)

    def load_model_(self, load_n):
        if self.stage == 'train':
            if self.train_phase == 'sr':
                net = self.model_sr
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module

                load_name = '{}_{}_{}_sr_{}.pt'.format(self.model_name, self.task, self.scale_param, load_n)
                load_path = os.path.join(self.save_dir, load_name)
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                net.load_state_dict(state_dict['model_sr_state_dict'], strict=False)  # ignore unexpected_kyes or missing_keys
            elif self.train_phase == 'fuse' and self.args.pre_train != '.':
                net1 = self.model_sr
                net2 = self.model_fuse
                if isinstance(net1, torch.nn.DataParallel):
                    net1 = net1.module
                if isinstance(net2, torch.nn.DataParallel):
                    net2 = net2.module
                load_name = '{}_{}_{}_{}.pt'.format(self.model_name, self.task, self.scale_param, load_n)
                load_path = os.path.join(self.save_dir, load_name)
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                net1.load_state_dict(state_dict['model_sr_state_dict'], strict=False)  # ignore unexpected_kyes or missing_keys
            elif self.train_phase == 'fuse' and self.args.pre_train == '.':
                net1 = self.model_sr
                net2 = self.model_fuse
                if isinstance(net1, torch.nn.DataParallel):
                    net1 = net1.module
                if isinstance(net2, torch.nn.DataParallel):
                    net2 = net2.module
                load_name = '{}_{}_{}_fuse_{}.pt'.format(self.model_name, self.task, self.scale_param, load_n)
                load_path = os.path.join(self.save_dir, load_name)
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                net1.load_state_dict(state_dict['model_sr_state_dict'], strict=False)  # ignore unexpected_kyes or missing_keys
                net2.load_state_dict(state_dict['model_fuse_state_dict'], strict=False)  # ignore unexpected_kyes or missing_keys
            else:
                if self.train_phase == 'joint' and self.args.pre_train == '.':
                    net1 = self.model_sr
                    net2 = self.model_fuse
                    if isinstance(net1, torch.nn.DataParallel):
                        net1 = net1.module
                    if isinstance(net2, torch.nn.DataParallel):
                        net2 = net2.module

                    load_name = '{}_{}_{}_joint_{}.pt'.format(self.model_name, self.task, self.scale_param, load_n)
                    load_path = os.path.join(self.save_dir, load_name)
                    print('loading the model from %s' % load_path)
                    state_dict = torch.load(load_path, map_location=str(self.device))
                    net1.load_state_dict(state_dict['model_sr_state_dict'],
                                        strict=False)  # ignore unexpected_kyes or missing_keys
                    net2.load_state_dict(state_dict['model_fuse_state_dict'],
                                         strict=False)  # ignore unexpected_kyes or missing_keys
                elif self.train_phase == 'joint' and self.args.pre_train != '.':
                    net1 = self.model_sr
                    net2 = self.model_fuse
                    if isinstance(net1, torch.nn.DataParallel):
                        net1 = net1.module
                    if isinstance(net2, torch.nn.DataParallel):
                        net2 = net2.module

                    load_name = '{}_{}_{}_{}.pt'.format(self.model_name, self.task, self.scale_param, load_n)
                    load_path = os.path.join(self.save_dir, load_name)
                    print('loading the model from %s' % load_path)
                    state_dict = torch.load(load_path, map_location=str(self.device))
                    net1.load_state_dict(state_dict['model_sr_state_dict'],
                                         strict=False)  # ignore unexpected_kyes or missing_keys
                    net2.load_state_dict(state_dict['model_fuse_state_dict'],
                                         strict=False)  # ignore unexpected_kyes or missing_keys
        elif self.stage == 'test':
            # net = self.model_sr
            # if isinstance(net, torch.nn.DataParallel):
            #     net = net.module
            #
            # load_name = '{}_{}_{}_{}.pt'.format(self.model_name, self.task, self.scale_param, self.test_final)
            # load_path = os.path.join(self.save_dir, load_name)
            # print('loading the model from %s' % load_path)
            # state_dict = torch.load(load_path, map_location=str(self.device))
            # net.load_state_dict(state_dict['model_sr_state_dict'], strict=False)  # ignore unexpected_kyes or missing_keys

            #-------------
            if self.train_phase == 'sr':
                net = self.model_sr
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module

                load_name = '{}_{}_{}_{}.pt'.format(self.model_name, self.task, self.scale_param, load_n)
                load_path = os.path.join(self.save_dir, load_name)
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                net.load_state_dict(state_dict['model_sr_state_dict'], strict=False)  # ignore unexpected_kyes or missing_keys
            elif self.train_phase == 'fuse' and self.args.pre_train != '.':
                net1 = self.model_sr
                net2 = self.model_fuse
                if isinstance(net1, torch.nn.DataParallel):
                    net1 = net1.module
                if isinstance(net2, torch.nn.DataParallel):
                    net2 = net2.module
                load_name = '{}_{}_{}_{}.pt'.format(self.model_name, self.task, self.scale_param, load_n)
                load_path = os.path.join(self.save_dir, load_name)
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                net1.load_state_dict(state_dict['model_sr_state_dict'], strict=False)  # ignore unexpected_kyes or missing_keys
            elif self.train_phase == 'fuse' and self.args.pre_train == '.':
                net1 = self.model_sr
                net2 = self.model_fuse
                if isinstance(net1, torch.nn.DataParallel):
                    net1 = net1.module
                if isinstance(net2, torch.nn.DataParallel):
                    net2 = net2.module
                load_name = '{}_{}_{}_{}.pt'.format(self.model_name, self.task, self.scale_param, load_n)
                load_path = os.path.join(self.save_dir, load_name)
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                net1.load_state_dict(state_dict['model_sr_state_dict'], strict=False)  # ignore unexpected_kyes or missing_keys
                net2.load_state_dict(state_dict['model_fuse_state_dict'], strict=False)  # ignore unexpected_kyes or missing_keys
            else:
                if self.train_phase == 'joint' and self.args.pre_train == '.':
                    net1 = self.model_sr
                    net2 = self.model_fuse
                    if isinstance(net1, torch.nn.DataParallel):
                        net1 = net1.module
                    if isinstance(net2, torch.nn.DataParallel):
                        net2 = net2.module

                    load_name = '{}_{}_{}_joint_{}.pt'.format(self.model_name, self.task, self.scale_param, load_n)
                    load_path = os.path.join(self.save_dir, load_name)
                    print('loading the model from %s' % load_path)
                    state_dict = torch.load(load_path, map_location=str(self.device))
                    net1.load_state_dict(state_dict['model_sr_state_dict'],
                                         strict=False)  # ignore unexpected_kyes or missing_keys
                    net2.load_state_dict(state_dict['model_fuse_state_dict'],
                                         strict=False)  # ignore unexpected_kyes or missing_keys
                elif self.train_phase == 'joint' and self.args.pre_train != '.':
                    net1 = self.model_sr
                    net2 = self.model_fuse
                    if isinstance(net1, torch.nn.DataParallel):
                        net1 = net1.module
                    if isinstance(net2, torch.nn.DataParallel):
                        net2 = net2.module

                    load_name = '{}_{}_{}_{}.pt'.format(self.model_name, self.task, self.scale_param, load_n)
                    load_path = os.path.join(self.save_dir, load_name)
                    print('loading the model from %s' % load_path)
                    state_dict = torch.load(load_path, map_location=str(self.device))
                    net1.load_state_dict(state_dict['model_sr_state_dict'],
                                         strict=False)  # ignore unexpected_kyes or missing_keys
                    net2.load_state_dict(state_dict['model_fuse_state_dict'],
                                         strict=False)  # ignore unexpected_kyes or missing_keys

    def set_mode(self, train):
        if train:
            self.model_sr.train()
            self.model_fuse.train()
        else:
            self.model_sr.eval()
            self.model_fuse.eval()



class CALayer(nn.Module):
    def __init__(self, channel, reduction = 16):
        super(CALayer, self).__init__()
        # adaptive avg pooling N x C x H x W --> N x C x 1 x 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, kernel_size=1, padding=0),  # in , out , kernel_size , padding
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction, channel, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x*y

class RCAB(nn.Module):
    def __init__(self, n_feat, ksize, reduction,
                 bias = True, bn = False, act = nn.ReLU(inplace=True), res_scale = 1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=ksize, padding=(ksize//2), bias= bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = 1

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class ResidualGroup(nn.Module):
    def __init__(self, n_feat, ksize, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [RCAB(n_feat, ksize, reduction) \
                        for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=ksize, padding=(ksize//2)))
        self.body = nn.Sequential(*modules_body)
    def forward(self, x):
        res = self.body(x)
        res += x
        return  res

class RCAN(nn.Module):
    def __init__(self, args):
        super(RCAN, self).__init__()

        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        ksize = 3
        reduction = args.reduction
        scale = args.scale

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, sign=-1)  #????????

        # #define head module
        modules_head = [nn.Conv2d(args.n_colors, n_feats, kernel_size=ksize, padding=(ksize//2))]

        # define body module
        modules_body = [ResidualGroup(n_feats, ksize, reduction, n_resblocks) \
                        for _ in range(n_resgroups)]
        modules_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=ksize, padding=(ksize//2)))

        # define tail module
        modules_tail = [Upsampler(scale, n_feats),
                        nn.Conv2d(n_feats, args.n_colors, kernel_size=ksize, padding=(ksize//2))]


        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, sign=1)  #???????????

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
    def forward(self, x):

        inputs = torch.chunk(x, 5, dim=1)
        outputs = []
        for input_ in inputs:

            x = self.head(input_)
            res = self.body(x)
            res += x
            y = self.tail(res)
            outputs.append(y)

        out = torch.cat(outputs, dim=1)

        return out



def default_conv(in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True, groups=1):
    if not padding and stride==1:
        padding = (kernel_size // 2)
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)

class ESA(nn.Module):
    def __init__(self, n_feats, conv=default_conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear')
        # c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bicubic')
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)

        return x * m


class ESARB(nn.Module):
    def __init__(self, n_feat, conv=default_conv, bn=False, kernel_size=3, bias=True):
        super(ESARB,self).__init__()
        m = []
        act=nn.ReLU(True)
        res_scale = 1
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)
        m.append(ESA(n_feat))
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        # res += x
        return res

class RFA(nn.Module):
    def __init__(self,n_feat,conv=default_conv):
        super(RFA, self).__init__()
        self.conv1 = conv(n_feat*4, n_feat, kernel_size=3)
        self.ESARB_1 = ESARB(n_feat)
        self.ESARB_2 = ESARB(n_feat)
        self.ESARB_3 = ESARB(n_feat)
        self.ESARB_4 = ESARB(n_feat)

    def forward(self,x):
        RB1 = self.ESARB_1(x)
        RB1_mid = x + RB1
        RB2 = self.ESARB_2(RB1_mid)
        RB2_mid = RB1_mid + RB2
        RB3 = self.ESARB_3(RB2_mid)
        RB3_mid = RB2_mid + RB3
        RB4 = self.ESARB_4(RB3_mid)
        out = [RB1,RB2,RB3,RB4]
        out = torch.cat(out,1)
        out_RB = self.conv1(out)
        return out_RB + x
#----------------------------------------------------------------------------------------checkpoints-3
class FuseB(nn.Module):
    def __init__(self,n_feat,conv=default_conv):
        super(FuseB,self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv1_1 = conv(n_feat,n_feat,kernel_size=3)
        self.conv1_2 = conv(n_feat,n_feat,kernel_size=3)
        self.conv1_3 = conv(n_feat,n_feat,kernel_size=3)
        self.conv1_4 = conv(n_feat,n_feat,kernel_size=3)
        self.conv1_5 = conv(n_feat,n_feat,kernel_size=3)

        self.Fconv1_1 = conv(n_feat*3,n_feat,kernel_size=3)
        self.Fconv1_2 = conv(n_feat*3,n_feat,kernel_size=3)
        self.Fconv1_3 = conv(n_feat*3,n_feat,kernel_size=3)

        self.Fconv2_1 = conv(n_feat*2,n_feat,kernel_size=3)
        self.Fconv2_2 = conv(n_feat*2,n_feat,kernel_size=3)
        self.Fconv2_3 = conv(n_feat*2,n_feat,kernel_size=3)
        self.Fconv2_4 = conv(n_feat*2,n_feat,kernel_size=3)
        self.Fconv2_5 = conv(n_feat*2,n_feat,kernel_size=3)

    def forward(self, output_list):
        input_0 = output_list[0]
        input_1 = output_list[1]
        input_2 = output_list[2]
        input_3 = output_list[3]
        input_4 = output_list[4]
        mid_0 = self.lrelu(self.conv1_1(input_0))
        mid_1 = self.lrelu(self.conv1_2(input_1))
        mid_2 = self.lrelu(self.conv1_3(input_2))
        mid_3 = self.lrelu(self.conv1_4(input_3))
        mid_4 = self.lrelu(self.conv1_5(input_4))
        frame_cat1 = torch.cat((mid_0,mid_1,mid_2),dim=1)
        frame_cat2 = torch.cat((mid_1,mid_2,mid_3),dim=1)
        frame_cat3 = torch.cat((mid_2,mid_3,mid_4),dim=1)
        frame_cat1 = self.Fconv1_1(frame_cat1)
        frame_cat2 = self.Fconv1_2(frame_cat2)
        frame_cat3 = self.Fconv1_3(frame_cat3)
        frame_0_cat = torch.cat((frame_cat1,mid_0),dim=1)
        frame_1_cat = torch.cat((frame_cat1,mid_1),dim=1)
        frame_2_cat = torch.cat((frame_cat2,mid_2),dim=1)
        frame_3_cat = torch.cat((frame_cat3,mid_3),dim=1)
        frame_4_cat = torch.cat((frame_cat3,mid_4),dim=1)
        out_0 = self.lrelu(self.Fconv2_1(frame_0_cat)) + input_0
        out_1 = self.lrelu(self.Fconv2_2(frame_1_cat)) + input_1
        out_2 = self.lrelu(self.Fconv2_3(frame_2_cat)) + input_2
        out_3 = self.lrelu(self.Fconv2_4(frame_3_cat)) + input_3
        out_4 = self.lrelu(self.Fconv2_5(frame_4_cat)) + input_4
        output_list = [out_0,out_1,out_2,out_3,out_4]
        return output_list

class RFANet(nn.Module):  #无共享参数checkpoint5
    def __init__(self, args):
        super(RFANet, self).__init__()

        n_resblocks = args.n_RFAs
        n_feats = args.n_feats
        ksize = 3
        scale = args.scale
        self.nonlocal_warp = NonLocalModule(kernel_size=5, n_colors=args.n_colors)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_feats*3, n_feats, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_feats*4, n_feats, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(n_feats*5, n_feats, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(n_feats*5, n_feats, kernel_size=3, padding=1)


        self.conv5 = nn.Conv2d(args.n_colors,n_feats//2,kernel_size=5,padding=2)
        self.conv6 = nn.Conv2d(args.n_colors,n_feats,kernel_size=5,padding=2)

        fuse_block = [FuseB(n_feats) for _ in range(12)]


        # #define head module
        modules_head = [nn.Conv2d(args.n_colors, n_feats, kernel_size=ksize, padding=(ksize//2))]

        # define body module
        modules_body = [RFA(n_feats) for _ in range(n_resblocks)]

        # define tail module
        modules_tail = [nn.Conv2d(n_feats, n_feats, kernel_size=ksize, padding=(ksize//2))]
        modules_tail.append(Upsampler(scale, n_feats))
        modules_tail.append(nn.Conv2d(n_feats, args.n_colors, kernel_size=ksize, padding=(ksize//2)))

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.fuse = nn.Sequential(*fuse_block)

    def forward(self, x):

        inputs = torch.chunk(x, 5, dim=1)
        k = len(inputs) // 2
        output_list2 = []
        for i in range(len(inputs)):
            # before_img = quantize(inputs[i],255)
            # save_image(before_img, '/home/vista/luZiTao/dataset/red_ICIG/test_red_noise_0/SR/B_{}.png'.format(i))

            # if i == k:
            #     output_ = inputs[k]
            #     # warp_img = quantize(output_, 255)
            #     # save_image(warp_img[0,:,:,:], '/home/vista/luZiTao/dataset/red_ICIG/test_red_noise_0/SR/{}.png'.format(i))
            # else:
            #     output_ = self.nonlocal_warp(inputs[i], inputs[k],shave=10, min_size=6400)      #torch 1,3,192,192

                # warp_img = quantize(output_, 255)
                # save_image(warp_img[0,:,:,:], '/home/vista/luZiTao/dataset/red_ICIG/test_red_noise_0/SR/{}.png'.format(i))
            output_ = self.nonlocal_warp(inputs[i], inputs[k],shave=10, min_size=6400)      #torch 1,3,192,192

            # warp_img = quantize(output_, 255)
            # save_image(warp_img[0,:,:,:], '/home/vista/luZiTao/dataset/red_ICIG/test_red_noise_0/SR/{}.png'.format(i))

            output_list2.append(output_)
        # out_frame3 = output_list2[2]


        frame1 = self.conv6(output_list2[0])
        frame2 = self.conv6(output_list2[1])
        frame3 = self.conv6(output_list2[2])
        frame4 = self.conv6(output_list2[3])
        frame5 = self.conv6(output_list2[4])
        output_list = [frame1,frame2,frame3,frame4,frame5]
        fuse_list = self.fuse(output_list)

        fuse_list_cat = torch.cat(fuse_list,dim=1)
        fuse_out = self.conv4(fuse_list_cat)
        # fuse_out = fuse_list[2]

        res = self.body(fuse_out)
        res += fuse_out
        y = self.tail(res)
        # y = y + inputs[2]
        return y

