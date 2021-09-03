import math
import cv2
import numpy as np
import torch
import torch.nn as nn
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from typing import List
from detectron2.structures import Boxes, ImageList, Instances

# from centernet.network.backbone import Backbone
from ..backbone import build_backbone
from ..losses import reg_l1_loss, modified_focal_loss, ignore_unlabel_focal_loss, mse_loss

from ..heads import ToyDetHead
from ..necks import FPNDeconv
from ..decoders import toydet_decode
from ..losses import BalanceL1Loss, mse_loss

from ..utils import batch_padding, mask_up_dim
from torch.nn import functional as F

__all__ = ["ToyDet"]

DEBUG = True
count = 0


@META_ARCH_REGISTRY.register()
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
        self.upsample = FPNDeconv(cfg)  # FPNDeconv(cfg)
        self.head = ToyDetHead(cfg)

        self.mean, self.std = cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(
            3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(
            3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.decoder = toydet_decode.ToyDetDecoder()

        self.to(self.device)

    def forward(self, batched_inputs):
        global count
        count += 1
        """
        Args:
            batched_inputs(list): batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
        """
        copy_imgs = batched_inputs.copy()

        file_name = batched_inputs[0]["file_name"]

        images = self.preprocess_image(batched_inputs)

        if not self.training:
            # return self.inference(images)
            rets = self.inference(images, batched_inputs)
            if not DEBUG:
                return rets

            for batch, result in zip(copy_imgs, rets):
                img = batch["image"]

                img = img.cpu().detach().numpy().transpose((1, 2, 0))

                #img = img[:, :, ::-1]
                img = img.astype(np.int8)

                # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                instances = result["instances"]
                rboxes = instances.rboxes

                for rbox in rboxes:
                    rbox = np.array(rbox, np.int)
                    img = cv2.polylines(img, [rbox],
                                        True,
                                        color=(0, 0, 155),
                                        thickness=2)
                # cv2.waitKey(0)

            return rets

        image_shape = images.tensor.shape[-2:]

        gt_segm = [x["sem_seg"].to(self.device) for x in batched_inputs]

        gt_segm = ImageList.from_tensors(gt_segm, 32, 0).tensor

        mask = [x["mask"].to(self.device) for x in batched_inputs]

        mask = ImageList.from_tensors(mask, 32, 0).tensor

        features = self.backbone(images.tensor)

        up_fmap = self.upsample(features)

        preds = self.head(up_fmap)

        if DEBUG and count % 10 == 0:

            img = copy_imgs[0]["image"]
            img = img.cpu().detach().numpy().transpose(
                (1, 2, 0)).astype(np.uint8)

            segm_show = (gt_segm[0].detach().cpu().numpy() * 255).astype(
                np.uint8)

            pred_show = (preds[0].detach().cpu().numpy() * 255).astype(
                np.uint8)[0]

            new_pred_show = mask_up_dim(pred_show)

            new_segm_show = mask_up_dim(segm_show, ratio=1)

            show = np.hstack((img, new_segm_show, new_pred_show))
            show = show.astype(np.uint8)
            #show = cv2.resize(show, (256*3, 256))
            font = cv2.FONT_HERSHEY_SIMPLEX

            show = cv2.putText(show, file_name, (20, 490), font, 0.7,
                               (255, 255, 255), 1)

            cv2.imshow("show", show)
            cv2.waitKey(2)

        loss_segm = self.head.losses(preds, gt_segm, mask)

        return loss_segm

    @torch.no_grad()
    def inference(self, images, batched_inputs, K=100):
        features = self.backbone(images.tensor)
        up_fmap = self.upsample(features)
        preds = self.head(up_fmap)

        # if DEBUG:
        #     pred_show = preds[0].detach().cpu().numpy()[0]
        #     cv2.imshow("det", (pred_show*255).astype(np.uint8))
        #     cv2.waitKey(2)

        scale_xys = [
            (batched_inputs[i]['width'] / float(images.image_sizes[i][1]),
             batched_inputs[i]['height'] / float(images.image_sizes[i][0]))
            for i in range(0, len(batched_inputs))
        ]

        results = []
        rbboxes_list, scores_list = self.decoder.decode_batch(
            scale_xys=scale_xys, heats=preds)
        for image_size, rbboxes, scores in zip(images.image_sizes,
                                               rbboxes_list, scores_list):

            result = Instances(image_size)

            result.rboxes = rbboxes
            result.scores = scores

            results.append({"instances": result})

        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(img / 255.) for img in images]
        images = ImageList.from_tensors(images, 32)
        return images

    @torch.no_grad()
    def inference_on_images(self, images: List, K=100, max_size=512):

        batch_images, params = self._preprocess(images, max_size)
        rets = self.inference(images, batch_images)

        return rets

    def _preprocess(self, images: List, max_size=512):
        """
        Normalize, pad and batch the input images.
        """
        batch_images = []
        params = []
        for image in images:
            old_size = image.shape[0:2]
            ratio = min(
                float(max_size) / (old_size[i]) for i in range(len(old_size)))
            new_size = tuple([int(i * ratio) for i in old_size])
            resize_image = cv2.resize(image, (new_size[1], new_size[0]))
            params.append({
                'width': old_size[1],
                'height': old_size[0],
                'resized_width': new_size[1],
                'resized_height': new_size[0]
            })
            batch_images.append(resize_image)
        batch_images = [
            torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
            for img in batch_images
        ]
        batch_images = [img.to(self.device) for img in batch_images]
        batch_images = [self.normalizer(img / 255.) for img in batch_images]
        batch_images = batch_padding(batch_images, 32)
        return batch_images, params


def build_toydet_model(cfg):
    model = ToyDet(cfg)
    return model
