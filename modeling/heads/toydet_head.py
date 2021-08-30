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
            cfg.MODEL.DETNET.NUM_CLASSES,
            bias_fill=True,
            bias_value=cfg.MODEL.DETNET.BIAS_VALUE,
        )
        self.ignore_value = -1
        self.loss_weight = 1.0
        self.common_stride = cfg.MODEL.DETNET.COMMON_STRIDE
        self.seg_head = SingleHead(64, 1)
        self.loss_func = BalanceL1Loss()

    def forward(self, x):
        segm = self.cls_head(x)
        segm = torch.sigmoid(segm)

        return segm

    def losses(self, predictions, targets, masks):
        targets = targets.unsqueeze(1)
        #print("tag here:",targets.shape)

        predictions = predictions.float()
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        cur_device = predictions.device
        targets = targets.to(cur_device)
        loss, loss_dict = self.loss_func(predictions, targets, masks)
        losses = {"loss_segm": loss * self.loss_weight}
        return losses
