import math
import numpy as np
import torch
import torch.nn as nn
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import Boxes, ImageList, Instances


from ..backbone import build_backbone
from ..decoders import crnn_decode
from .crnn_model import get_crnn
# from ..losses import ctc_loss
# from str_label_converter import strLabelConverter
from data.dataset.recognizer_utils.get_360cc_labels import get_360cc_labels
from data.dataset.recognizer_utils.str_label_converter import strLabelConverter
import data.dataset.recognizer_utils.alphabets as alphabets

__all__ = ["CRNNet"]


@META_ARCH_REGISTRY.register()
class CRNNet(nn.Module):
    """
    Implement CRNNet
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.cfg = cfg
        # cfg.DATASETS.ALPHABETS = \
        alphabet = alphabets.alphabet
        num_classes = len(alphabet)
        print("num_class in crnn:", num_classes)
        # cfg.MODEL.CRNN.NUM_CLASSES = len(cfg.DATASETS.ALPHABETS)
        self.model = get_crnn(cfg, num_classes)
        # self.backbone = build_backbone(cfg)

        # crnn_in_channels = cfg.MODEL.CRNN.IN_CHANNELS
        # self.alphabet = cfg.MODEL.ALPHABET
        # self.num_classes = len(self.alphabet) + 1
        # self.crnn_decode = crnn_decode.CRNNDecoder(crnn_in_channels, self.num_classes)
        self.loss_func = torch.nn.CTCLoss()
        self.labels = get_360cc_labels(cfg, True)
        self.converter = strLabelConverter(cfg.DATASETS.ALPHABETS)
        self.to(self.device)

    def forward(self, batched_inputs):
        # image, text, length = self.preprocess_image(batched_inputs)
        inp, idx = batched_inputs
        labels = self.get_batch_label(idx)  # length = 一个batch中的总字符长度, text = 一个batch中的字符所对应的下标
        if not self.training:
            # return self.inference(images)
            return self.inference(inp)

        # image_shape = images.tensor.shape[-2:]

        # features = self.backbone(inp)

        # features = features[self.cfg.MODEL.RESNETS.OUT_FEATURES[0]]
        # preds = self.crnn_decode(features)
        preds = self.model(inp)
        batch_size = inp.size(0)
        text, length = self.converter.encode(labels)  # length = 一个batch中的总字符长度, text = 一个batch中的字符所对应的下标
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)  # timestep * batchsize

        loss = self.loss_func(preds, text, preds_size, length)


        gt_loss = {"loss_ctc": loss}
        loss = {**loss, **gt_loss}
        return loss

    @torch.no_grad()
    def inference(self, image):
        features = self.backbone(image.tensor)
        preds = self.crnn_decode(features)
        batch_size = self.cfg.SOLVER.IMS_PER_BATCH
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        _, preds = preds.max(2)
        preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = self.converter.decode(preds.data, preds_size.data, raw=False)
        raw_preds = self.converter.decode(preds.data, preds_size.data, raw=True)[:self.cfg.TEST.N_TEST_DISP]

        return sim_preds, raw_preds

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        batch_size = self.cfg.SOLVER.IMS_PER_BATCH

        image = torch.FloatTensor(batch_size, 3, self.cfg.INPUT.IMG_W, self.cfg.INPUT.IMG_H)
        image = image.to(self.device)
        text = torch.IntTensor(batch_size * 5)
        length = torch.IntTensor(batch_size)
        cpu_images, cpu_texts = batched_inputs
        batch_size = cpu_images.size(0)
        self.loadData(image, cpu_images)
        t, l = self.converter.encode(cpu_texts)

        self.loadData(text, t)
        self.loadData(length, l)

        return image, text, length

    def loadData(self, v, data):
        v.data.resize_(data.size()).copy_(data)

    def get_batch_label(self, i):
        label = []
        for idx in i:
            label.append(list(self.labels[idx].values())[0])
        return label