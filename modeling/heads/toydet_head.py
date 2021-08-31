import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from ..losses import BalanceL1Loss


class SingleHead(nn.Module):
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
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """

    def __init__(self, cfg, ignore_value=-1):
        super(ToyDetHead, self).__init__()
        self.cls_head = SingleHead(
            64,
            1,
            bias_fill=True,
            bias_value=cfg.MODEL.DETNET.BIAS_VALUE,
        )
        self.ignore_value = -1
        self.loss_weight = 1.0
        self.common_stride = cfg.MODEL.DETNET.COMMON_STRIDE

        self.loss_func = nn.MSELoss(reduce='none')
        self.balanced_mse = BalanceL1Loss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.cls_head(x)
        x = self.sigmoid(x)

        return x

    def losses(self, predictions, targets, masks):

        #print("tag here:",targets.shape)

        dst_shape = (targets.size(1), targets.size(2))
        # targets = targets.to(cur_device)

        predictions = F.upsample(
            input=predictions, size=dst_shape, mode='bilinear')

        # targets = targets.unsqueeze(0)
        # targets = F.interpolate(
        #     targets, size=dst_shape, mode="bilinear", align_corners=False
        # )
        # targets = targets.squeeze(0)

        # masks = masks.unsqueeze(0)
        # masks = F.interpolate(
        #     masks, size=dst_shape, mode="bilinear", align_corners=False
        # )
        # masks = masks.squeeze(0)

        # predictions=predictions.squeeze(0)
        loss = 100*(self.loss_func(predictions, targets)
                   * masks).sum()/masks.sum()

        return dict(loss=loss)
