from configs import add_textnet_config, add_basic_config
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
import cv2

import torch
import torch.nn as nn
import sys
import os
from tqdm import tqdm
from utils import create_text_labels, batch_padding
import numpy as np
from typing import List

from modeling.decoders import toydet_decode
from modeling.backbone import build_backbone
from modeling.heads import *
from modeling.necks import *
from modeling.decoders import *

sys.path.insert(0, '.')


class ToyDet(nn.Module):
    """
    Implement ToyDet
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.cfg = cfg

        # fmt: off
        self.num_classes = cfg.MODEL.DETNET.NUM_CLASSES
        # Loss parameters:
        # Inference parameters:
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # fmt: on
        self.backbone = build_backbone(cfg)
        self.upsample = FPNDeconv(cfg)
        self.head = ToyDetHead(cfg)

        self.mean, self.std = cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(
            self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(
            self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.decoder = toydet_decode.ToyDetDecoder()

        self.to(self.device)

    @torch.no_grad()
    def _forward(self, batch_images):

        features = self.backbone(batch_images)
        up_fmap = self.upsample(features)
        preds = self.head(up_fmap)

        return preds

    @torch.no_grad()
    def inference_on_images(self, images: List, K=100, max_size=512):

        batch_images, batched_inputs = self._preprocess(images, max_size)
        # images=self.preprocess_image(batch_images)

        preds = self._forward(batch_images=batch_images)

        scale_xys= [(batched_inputs[i]['width'] / float(batched_inputs[i]['resized_width']), 
                batched_inputs[i]['height'] / float(batched_inputs[i]['resized_height'])) for i in range(0,len(batched_inputs))]
        
        rets = self.decoder.decode_batch(scale_xys=scale_xys, heats=preds)

        return rets

    def _preprocess(self, images: List, max_size=512):
        """
        Normalize, pad and batch the input images.
        """
        batch_images = []
        params = []
        for image in images:
            old_size = image.shape[0:2]
            ratio = min(float(max_size) / (old_size[i])
                        for i in range(len(old_size)))
            new_size = tuple([int(i * ratio) for i in old_size])
            resize_image = cv2.resize(image, (new_size[1], new_size[0]))
            params.append({'width': old_size[1],
                           'height': old_size[0],
                           'resized_width': new_size[1],
                           'resized_height': new_size[0]
                           })
            batch_images.append(resize_image)
        batch_images = [torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
                        for img in batch_images]
        batch_images = [img.to(self.device) for img in batch_images]
        batch_images = [self.normalizer(img/255.) for img in batch_images]
        batch_images = batch_padding(batch_images, 32)
        return batch_images, params


def build_model(cfg):

    model = ToyDet(cfg)
    return model


if __name__ == "__main__":
    # cfg
    cfg = get_cfg()
    add_textnet_config(cfg)
    cfg.merge_from_file("yamls/text_detection/toydet_text.yaml")

    # model
    model = build_model(cfg)
    DetectionCheckpointer(model).load(
        "exp_results/toydet_exp_R50/model_final.pth")
    model.eval()

    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    )

    class_names = ('text')
    # txt
    txt = open('results.txt', 'w')

    out_dir = 'exp_results/toydet_exp_R50/results/'
    # os.mkdir(out_dir)
    thickness = 1
    bbox_color = (0, 255, 0)
    text_color = bbox_color
    font_scale = 0.5
    # images
    #lines = open('datasets/bjz_multicameras_20200615/txts/bjz_val_multicameras_20200615.txt').readlines()
    root = 'datasets/icdar/images/'
    images = [root + i for i in sorted(os.listdir(root))]
    images = images[0:9]
    print(images)
    bs = 1
    from .text_eval import TextEvaluator
from .text_eval_scripts import text_eval_main
from . import rrc_evaluation_funcss,list):
                continue 
            
            scores = scores_list[k]
             
            H, W, C = images_rgb[k].shape
            img = images_rgb[k][:, :, ::-1]
            img_name = img_names[k]
            outstr=""
            for box,score in zip(boxes, scores):
                
                box=box.astype(np.int32)
                print(box.shape)
                
                for i in range(len(box)):
                    outstr = outstr + str(int(box[i][0])) +','+str(int(box[i][1])) +','

                #box = box.astype(np.int32).reshape((-1, 1, 2))
                
                
                cv2.polylines(img, [box], True, color=(255, 255, 155), thickness=2)
                # cv2.putText(img, "text", (box[0]),
                #             cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
             
            cv2.imshow("hello",cv2.resize(img,(int(W/2),int(H/2))))
            cv2.waitKey(0)
            # cv2.imwrite(os.path.join(out_dir, img_name), img)
