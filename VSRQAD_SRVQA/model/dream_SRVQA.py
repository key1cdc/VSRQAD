from model.basemodel import BaseModel
import torch
from utility import make_optimizer1, make_scheduler1,MeanShift,Upsampler,make_optimizer2,make_scheduler2
import os
from torchvision import models
import torch.nn as nn
from model import common
from torch import Tensor
# from .._internally_replaced_utils import load_state_dict_from_url
from basic import load_state_dict_from_url

from typing import Type, Any, Callable, Union, List, Optional

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)



class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet_ST(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet_ST, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv1_y = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv1_out = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0,
                                 bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1)
        self.fc_resnet = nn.Sequential(
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # sns.set()
        x = self.conv1(x)  # 320->160
        # heat1 = torch.mean(x[0, :, :, :], dim=0)
        # ax = sns.heatmap(heat1.detach().numpy())
        # heat1 = torch.mean(x[0, :, :, :], dim=0)
        # ax = sns.heatmap(heat1.detach().numpy())
        x = self.bn1(x)
        x = self.relu(x)
        # y = self.bn1(y)
        # y = self.relu(y)
        x = self.maxpool(x)  # 80
        # y = self.maxpool(y)
        x = self.layer1(x)  # 80
        x = self.layer2(x)  # 40
        # heat1 = torch.mean(x[0, :, :, :], dim=0)
        # ax = sns.heatmap(heat1.detach().numpy())
        x = self.layer3(x)  # 20
        x = self.layer4(x)  # 10
        x = self.conv1_out(x)
        ave = self.avgpool(x)
        std = torch.std(x.view(x.size()[0], x.size()[1], -1), dim=2, keepdim=False)
        ave = torch.flatten(ave, 1)
        std = torch.flatten(std, 1)
        x = torch.cat((ave, std),dim=1)
        #print('T', x.shape)
        # x = self.fc(x)

        return ave

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv1_y = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1)
        self.fc_resnet = nn.Sequential(
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, y: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # sns.set()
        x = self.conv1(x)  # 320->160
        y = self.conv1_y(y)
        # heat1 = torch.mean(x[0, :, :, :], dim=0)
        # ax = sns.heatmap(heat1.detach().numpy())
        x = x * (1 + abs(y) * 2)
        # heat1 = torch.mean(x[0, :, :, :], dim=0)
        # ax = sns.heatmap(heat1.detach().numpy())
        x = self.bn1(x)
        x = self.relu(x)
        # y = self.bn1(y)
        # y = self.relu(y)
        x = self.maxpool(x)  # 80
        # y = self.maxpool(y)
        x = self.layer1(x)  # 80
        x = self.layer2(x)  # 40
        # heat1 = torch.mean(x[0, :, :, :], dim=0)
        # ax = sns.heatmap(heat1.detach().numpy())
        x = self.layer3(x)  # 20
        x = self.layer4(x)  # 10
        ave = self.avgpool(x)
        std = torch.std(x.view(x.size()[0], x.size()[1], -1), dim=2, keepdim=False)
        ave = torch.flatten(ave, 1)
        std = torch.flatten(std, 1)
        x = torch.cat((ave, std), dim=1)
        #print('S', x.shape)
        # x = self.fc(x)

        return ave

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self._forward_impl(x, y)

def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict,strict=False)
    return model

def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def _resnet2(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet_ST:
    model = ResNet_ST(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict,strict=False)
    return model

def resnet18_(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet_ST:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet2('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


class SpatialNet3(nn.Module):
    def __init__(self, args):
        super(SpatialNet3, self).__init__()
        num_classes = 3
        kernel_sizes = 3
        act = nn.ReLU(True)
        n_feats = args.n_feats
        conv = common.default_conv
        self.pre_resnet = resnet18(pretrained=False)
        self.ST_resnet = resnet18_(pretrained=False)
        # self.model_fea = nn.Sequential(*list(self.pre_resnet.children())[:-1])
        self._norm_layer = nn.BatchNorm2d
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.avgpool = nn.AvgPool2d((6, 6))
        self.conv_in1_1 = conv(3, n_feats, 7)
        self.tanh = nn.Tanh()
        self.conv_f_1 = nn.Conv2d(n_feats*16, 512, kernel_size=(3, 3), stride=(1, 1), padding=0)

        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.fc_resnext = nn.Sequential(
           # nn.Linear(2048, 512),
          #  nn.BatchNorm1d(512),
           # nn.Dropout(0.1),
          #  nn.ReLU(),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.fc2 = nn.Linear(1152, 1)

        self.fc_ST = nn.Sequential(
            nn.Linear(512+256, 256),
         #   nn.LayerNorm(256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.fc_ST_B = nn.Linear(512*64,1)

    def forward(self, x, y, z):
        # X = torch.cat((x, y), dim=1)
        X = self.pre_resnet(x, y)
        Z = self.ST_resnet(z)
        #X = X.unsqueeze(2)
        #Z = Z.unsqueeze(2)
        #ST = torch.bmm(X, torch.transpose(Z, 1, 2))
        #ST = ST.view(x.size()[0], 512*256)
        ST = torch.cat((X, Z), dim=1)
       # print(ST.shape)
        X = self.fc_ST(ST)
        return X.squeeze_(1)



class dream_SRVQA(BaseModel):
    def initialize(self, args):
        BaseModel.initialize(self, args)
        self.spatial_model = SpatialNet3(args)
        self.spatial_model = nn.DataParallel(self.spatial_model, device_ids=args.gpu_ids)
        self.spatial_model.to(self.device)

        self.criterion = self.loss_define()
        self.n_colors = args.n_colors
        if self.train_phase == 'sr':
            self.optimizer1 = make_optimizer1(args, self.spatial_model.parameters())
            self.scheduler1 = make_scheduler1(args,self.optimizer1)

    def train(self):
        self.out1 = self.spatial_model(self.sr, self.sr-self.lr, self.st)
        loss_sum = 0
        for (loss_type, loss_fun) in self.criterion.items():
            if loss_type == 'L1':
                for i in range(len(self.mos)):
                    if self.mos.float()[i] > 0.9:
                        mul = 21.32
                    elif self.mos.float()[i] > 0.8:
                        mul = 6.28
                    elif self.mos.float()[i] > 0.7:
                        mul = 5.01
                    elif self.mos.float()[i] > 0.6:
                        mul = 6.57
                    elif self.mos.float()[i] > 0.5:
                        mul = 8.19
                    elif self.mos.float()[i] > 0.4:
                        mul = 10.04
                    elif self.mos.float()[i] > 0.3:
                        mul = 13.37
                    elif self.mos.float()[i] > 0.2:
                        mul = 14.67
                    elif self.mos.float()[i] > 0.1:
                        mul = 28.61
                    else:
                        mul = 23.54
                    loss_sum += loss_fun(self.mos.float()[i], self.out1.float()[i]) * mul
                # loss1 = loss_fun(self.mos.float(), self.out1.float())
            # if loss_type == 'Cross':
                # loss2 = loss_fun(self.out2, self.scale)
        loss = loss_sum/len(self.mos)
        losses = {'LOSS': loss}
        #print(self.optimizer1.state_dict()['param_groups'][0]['lr'])
        self.optimizer1.zero_grad()
        loss.backward()
        self.optimizer1.step()
        return losses

    def eval(self):
        with torch.no_grad():
            self.out1 = self.spatial_model(self.sr, self.sr-self.lr, self.st)
            loss_sum = 0
            for (loss_type, loss_fun) in self.criterion.items():
                if loss_type == 'L1':
                    for i in range(len(self.mos)):
                        if self.mos.float()[i] > 0.9:
                            mul = 21.32
                        elif self.mos.float()[i] > 0.8:
                            mul = 6.28
                        elif self.mos.float()[i] > 0.7:
                            mul = 5.01
                        elif self.mos.float()[i] > 0.6:
                            mul = 6.57
                        elif self.mos.float()[i] > 0.5:
                            mul = 8.19
                        elif self.mos.float()[i] > 0.4:
                            mul = 10.04
                        elif self.mos.float()[i] > 0.3:
                            mul = 13.37
                        elif self.mos.float()[i] > 0.2:
                            mul = 14.67
                        elif self.mos.float()[i] > 0.1:
                            mul = 28.61
                        else:
                            mul = 23.54
                        loss_sum += loss_fun(self.mos.float()[i], self.out1.float()[i]) * mul
                    # loss1 = loss_fun(self.mos.float(), self.out1.float())
            loss = loss_sum/len(self.mos)
        return self.out1, self.mos, loss.item()

    def test(self):
        with torch.no_grad():
            self.out1 = self.spatial_model(self.sr, self.sr-self.lr, self.st)

        return self.out1

    def set_mode(self, train):
        if train:
            self.spatial_model.train()
        else:
            self.spatial_model.eval()

    def set_input(self, input):
        self.lr = input['lr'].to(self.device)
        self.sr = input['sr'].to(self.device)
        self.st = input['st'].to(self.device)
        self.mos = input['mos'].to(self.device)
        self.scale = input['scale'].to(self.device)

    def set_eval_input(self, input):
        self.lr = input['lr'].to(self.device)
        self.sr = input['sr'].to(self.device)
        self.st = input['st'].to(self.device)
        self.mos = input['mos'].to(self.device)
        self.scale = input['scale'].to(self.device)

    def set_test_input(self, input):
        self.lr = input['lr'].to(self.device)
        self.sr = input['sr'].to(self.device)
        self.st = input['st'].to(self.device)

    def get_results(self):
        images = {'lr': self.lr, 'output': self.s_out,
                  'hr': self.hr, 'sr': self.sr}

    def save_model(self, is_best=False):
        save_name = '{}'.format(self.model_name)
        net = self.spatial_model
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            if is_best:
                save_name_ = save_name + '_epoch{}_best.pt'.format(self.epoch)
                torch.save(
                    net.module.cpu().state_dict(),
                    os.path.join(self.save_dir, save_name_),
                    _use_new_zipfile_serialization=False
                )
                print(save_name)

            save_name_ = save_name + 'epoch{}.pt'.format(self.epoch)
            # torch.save(
            #     net.module.cpu().state_dict(),
            #     os.path.join(self.save_dir, save_name_),
            #     _use_new_zipfile_serialization=False
            # )
            # print(save_name)
            net.cuda(self.gpu_ids[0])

        else:
            torch.save(net.cpu().state_dict(), os.path.join(self.save_dir, save_name),_use_new_zipfile_serialization=False)
            if is_best:
                save_name_ = save_name + 'best.pt'
                torch.save(net.cpu().state_dict(), os.path.join(self.save_dir, save_name_),_use_new_zipfile_serialization=False)

            save_name_ = save_name + 'epoch{}.pt'.format(self.epoch)
            torch.save(net.cpu().state_dict(), os.path.join(self.save_dir, save_name_),_use_new_zipfile_serialization=False)
            print(save_name)

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

    def load_test_model(self):
        self.load_model_('epoch{}'.format(self.args.resume))

    def load_model_(self, load_n):
        net = self.spatial_model
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        load_name = '{}{}.pt'.format(self.model_name, load_n)
        load_path = os.path.join(self.save_dir, load_name)
        print('loading the model from %s' % load_path)

        state_dict = torch.load(load_path, map_location=str(self.device))
        net.load_state_dict(state_dict, strict=False)   # ignore unexpected_kyes or missing_keys
