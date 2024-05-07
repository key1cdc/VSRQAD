
import os
import math
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn as nn
import time

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_optimizer1(args, my_model):    #####
    # trainable = filter(lambda x: x.requires_grad,
    #                    my_model.parameters())  # choose the requires_grad=True paramenters to update.

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(my_model, **kwargs)
def make_optimizer2(args, my_model):    #####
    # trainable = filter(lambda x: x.requires_grad,
    #                    my_model.parameters())  # choose the requires_grad=True paramenters to update.

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr2
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(my_model, **kwargs)

def make_scheduler1(args, my_optimizer1):
    if args.decay_type == 'step':
        scheduler1 = lrs.StepLR(
            my_optimizer1,
            step_size=args.lr_decay,
            gamma=args.gamma
        )

    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler1 = lrs.MultiStepLR(
            my_optimizer1,
            milestones=milestones,
            gamma=args.gamma
        )

    return scheduler1

def make_scheduler2(args, my_optimizer2):
    if args.decay_type == 'step':

        scheduler2 = lrs.StepLR(
            my_optimizer2,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler2 = lrs.MultiStepLR(
            my_optimizer2,
            milestones=milestones,
            gamma=args.gamma
        )

    return scheduler2

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

#
class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feat, 4 * n_feat, kernel_size= 3, padding=1,  bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(nn.Conv2d(n_feat, 9 * n_feat, kernel_size= 3, padding= 1, bias = bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)