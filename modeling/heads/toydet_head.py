import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from ..losses import BalanceL1Loss


class SingleHead(nn.Module):
    """
    The single head used in different modules

    """

    def __init__(self, in_channel, out_channel, bias_fill=False, bias_value=0):
        super(SingleHead, self).__init__()
        self.feat_conv = nn.Conv2d(
            in_channel, in_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.out_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        if bias_fill:
            self.out_conv.bias.data.fill_(bias_value)
        self.feat_conv.apply(self.weights_init)
        self.out_conv.apply(self.weights_init)
        self.relu.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, x):
        x = self.feat_conv(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x


class ToyDetHead(nn.Module):
    """
    The head used in ToyDet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """

    def __init__(self, cfg, ignore_value=-1, inner_channels=256, bias=False, out_channel=1):
        super(ToyDetHead, self).__init__()

        self.inner_channels = inner_channels

        self.cls = nn.Sequential(
            nn.Conv2d(self.inner_channels, self.inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(self.inner_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inner_channels//4, self.inner_channels //
                      16, 3, padding=1, bias=bias),
            nn.BatchNorm2d(self.inner_channels//16),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inner_channels//16, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid())

        self.cls.apply(self.weights_init)

        self.ignore_value = -1
        self.loss_weight = 1.0

        self.common_stride = cfg.MODEL.DETNET.COMMON_STRIDE

        self.loss_func = nn.MSELoss(reduce='none')
        self.balanced_mse = BalanceL1Loss()
        self.loss = nn.MSELoss()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, x):
        x = self.cls(x)

        return x

    def losses(self, predictions, targets, masks):

        dst_shape = (targets.size(1), targets.size(2))

        predictions = predictions.float()

        predictions = F.upsample(
             input=predictions, size=dst_shape, mode='bilinear')

        #predictions = predictions.squeeze(1)
        loss,_=self.balanced_mse(predictions,targets,masks)
        # print(predictions.shape,targets.shape)
        #loss =self.loss(predictions*masks,targets*masks)
        # print(predictions.shape, targets.shape, masks.shape)
        # diff2 = (torch.flatten(predictions) - torch.flatten(targets)) ** 2.0 * torch.flatten(masks)
        # loss = torch.sum(diff2) / torch.sum(masks)

        return dict(loss=loss)
