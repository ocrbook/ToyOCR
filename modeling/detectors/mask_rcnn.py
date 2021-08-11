import cv2
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.config import configurable
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.structures import ImageList, Instances
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads


__all__ = ["OcrMaskRCNN"]


@META_ARCH_REGISTRY.register()
class OcrMaskRCNN(GeneralizedRCNN):
    """
    OCR Mask-RCNN,Any models that contains the following three components:
    1.featreu extraction
    2.Region proposal generation
    3.Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        cfg
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__(self,cfg)
        
        
    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        ret = super().inference()
        return ret 

    def ocr_post_process(self, ret):
        return ret
    
    
    @classmethod
    def from_config(cls, cfg):
        print("dadyishere:",cfg)
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }
