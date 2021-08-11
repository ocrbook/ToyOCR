# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_textnet_config(cfg):
    """
    Add config for tridentnet.
    """
    _C = cfg

    # DETNET config
    _C.MODEL.DETNET = CN()
    _C.MODEL.DETNET.DECONV_CHANNEL = [512, 256, 128, 64]
    _C.MODEL.DETNET.DECONV_KERNEL = [4, 4, 4]
    _C.MODEL.DETNET.NUM_CLASSES = 80
    _C.MODEL.DETNET.COMMON_STRIDE = 4
    _C.MODEL.DETNET.MODULATE_DEFORM = True
    _C.MODEL.DETNET.USE_DEFORM = True
    _C.MODEL.DETNET.BIAS_VALUE = -2.19
    _C.MODEL.DETNET.DOWN_SCALE = 4
    _C.MODEL.DETNET.MIN_OVERLAP = 0.7
    _C.MODEL.DETNET.TENSOR_DIM = 128
    _C.MODEL.DETNET.OUTPUT_SIZE = [128, 128]
    _C.MODEL.DETNET.BOX_MINSIZE = 1e-5
    _C.MODEL.DETNET.TRAIN_PIPELINES = [
        # ("CenterAffine", dict(boarder=128, output_size=(512, 512), random_aug=True)),
        ("RandomFlip", dict()),
        ("RandomBrightness", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomContrast", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomSaturation", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomLighting", dict(scale=0.1)),
    ]
    _C.MODEL.DETNET.TEST_PIPELINES = []
    _C.MODEL.DETNET.LOSS = CN()
    _C.MODEL.DETNET.LOSS.CLS_WEIGHT = 1
    _C.MODEL.DETNET.LOSS.WH_WEIGHT = 0.1
    _C.MODEL.DETNET.LOSS.REG_WEIGHT = 1
    _C.MODEL.DETNET.LOSS.NORM_WH = False
    _C.MODEL.DETNET.LOSS.SKIP_LOSS = False
    _C.MODEL.DETNET.LOSS.SKIP_WEIGHT = 1.0
    _C.MODEL.DETNET.LOSS.MSE = False
    _C.MODEL.DETNET.LOSS.IGNORE_UNLABEL = False

    _C.MODEL.DETNET.LOSS.COMMUNISM = CN()
    _C.MODEL.DETNET.LOSS.COMMUNISM.ENABLE = False
    _C.MODEL.DETNET.LOSS.COMMUNISM.CLS_LOSS = 1.5
    _C.MODEL.DETNET.LOSS.COMMUNISM.WH_LOSS = 0.3
    _C.MODEL.DETNET.LOSS.COMMUNISM.OFF_LOSS = 0.1

    _C.MODEL.DETNET.IMGAUG_PROB = 2.0

    # rewrite backbone
    _C.MODEL.BACKBONE = CN()
    _C.MODEL.BACKBONE.NAME = "build_resnet"
    _C.MODEL.BACKBONE.DEPTH = 18
    _C.MODEL.BACKBONE.STR_DEPTH = "400MF"
    _C.MODEL.BACKBONE.LAST_STRIDE = 2
    _C.MODEL.BACKBONE.PRETRAIN_PATH = ""
    _C.MODEL.BACKBONE.PRETRAIN = True
    _C.MODEL.BN_TYPE = "SyncBN"

    # optim and min_lr(for cosine schedule)
    _C.SOLVER.MIN_LR = 1e-8
    _C.SOLVER.OPTIM_NAME = "SGD"
    _C.SOLVER.COSINE_DECAY_ITER = 0.7
    _C.SOLVER.REFERENCE_WORLD_SIZE = 0

    # SWA options
    _C.SOLVER.SWA = CN()
    _C.SOLVER.SWA.ENABLED = False
    _C.SOLVER.SWA.ITER = 10
    _C.SOLVER.SWA.PERIOD = 2
    _C.SOLVER.SWA.LR_START = 2.5e-6
    _C.SOLVER.SWA.ETA_MIN_LR = 3.5e-6
    _C.SOLVER.SWA.LR_SCHED = False

    # input config
    _C.INPUT.FORMAT = "RGB"
    _C.INPUT.RESIZE_TYPE = "ResizeShortestEdge"


def add_centernet_config(cfg):
    """
    Add config for tridentnet.
    """
    _C = cfg

    # centernet config
    _C.MODEL.DETNET = CN()
    _C.MODEL.DETNET.DECONV_CHANNEL = [512, 256, 128, 64]
    _C.MODEL.DETNET.DECONV_KERNEL = [4, 4, 4]
    _C.MODEL.DETNET.NUM_CLASSES = 5
    _C.MODEL.DETNET.MODULATE_DEFORM = True
    _C.MODEL.DETNET.USE_DEFORM = True
    _C.MODEL.DETNET.BIAS_VALUE = -2.19
    _C.MODEL.DETNET.DOWN_SCALE = 4
    _C.MODEL.DETNET.MIN_OVERLAP = 0.7
    _C.MODEL.DETNET.TENSOR_DIM = 128
    _C.MODEL.DETNET.OUTPUT_SIZE = [128, 128]
    _C.MODEL.DETNET.BOX_MINSIZE = 1e-5
    _C.MODEL.DETNET.TRAIN_PIPELINES = [
        # ("CenterAffine", dict(boarder=128, output_size=(512, 512), random_aug=True)),
        ("RandomFlip", dict()),
        ("RandomBrightness", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomContrast", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomSaturation", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomLighting", dict(scale=0.1)),
    ]
    _C.MODEL.DETNET.TEST_PIPELINES = []
    _C.MODEL.DETNET.LOSS = CN()
    _C.MODEL.DETNET.LOSS.CLS_WEIGHT = 1
    _C.MODEL.DETNET.LOSS.WH_WEIGHT = 0.1
    _C.MODEL.DETNET.LOSS.REG_WEIGHT = 1
    _C.MODEL.DETNET.LOSS.NORM_WH = False
    _C.MODEL.DETNET.LOSS.SKIP_LOSS = False
    _C.MODEL.DETNET.LOSS.SKIP_WEIGHT = 1.0
    _C.MODEL.DETNET.LOSS.MSE = False
    _C.MODEL.DETNET.LOSS.IGNORE_UNLABEL = False

    _C.MODEL.DETNET.LOSS.COMMUNISM = CN()
    _C.MODEL.DETNET.LOSS.COMMUNISM.ENABLE = False
    _C.MODEL.DETNET.LOSS.COMMUNISM.CLS_LOSS = 1.5
    _C.MODEL.DETNET.LOSS.COMMUNISM.WH_LOSS = 0.3
    _C.MODEL.DETNET.LOSS.COMMUNISM.OFF_LOSS = 0.1

    _C.MODEL.DETNET.IMGAUG_PROB = 2.0

    # rewrite backbone
    _C.MODEL.BACKBONE = CN()
    _C.MODEL.BACKBONE.NAME = "build_resnet"
    _C.MODEL.BACKBONE.DEPTH = 18
    _C.MODEL.BACKBONE.STR_DEPTH = "400MF"
    _C.MODEL.BACKBONE.LAST_STRIDE = 2
    _C.MODEL.BACKBONE.PRETRAIN_PATH = ""
    _C.MODEL.BACKBONE.PRETRAIN = True
    _C.MODEL.BN_TYPE = "SyncBN"

    # optim and min_lr(for cosine schedule)
    _C.SOLVER.MIN_LR = 1e-8
    _C.SOLVER.OPTIM_NAME = "SGD"
    _C.SOLVER.COSINE_DECAY_ITER = 0.7

    # SWA options
    _C.SOLVER.SWA = CN()
    _C.SOLVER.SWA.ENABLED = False
    _C.SOLVER.SWA.ITER = 10
    _C.SOLVER.SWA.PERIOD = 2
    _C.SOLVER.SWA.LR_START = 2.5e-6
    _C.SOLVER.SWA.ETA_MIN_LR = 3.5e-6
    _C.SOLVER.SWA.LR_SCHED = False

    _C.SOLVER.REFERENCE_WORLD_SIZE = 0

    # input config
    _C.INPUT.FORMAT = "RGB"
    _C.INPUT.RESIZE_TYPE = "ResizeShortestEdge"


def add_basic_config(cfg):
    # optim and min_lr(for cosine schedule)
    _C = cfg

    _C.SOLVER.MIN_LR = 1e-8
    _C.SOLVER.OPTIM_NAME = "SGD"
    _C.SOLVER.COSINE_DECAY_ITER = 0.7

    # SWA options
    _C.SOLVER.SWA = CN()
    _C.SOLVER.SWA.ENABLED = False
    _C.SOLVER.SWA.ITER = 10
    _C.SOLVER.SWA.PERIOD = 2
    _C.SOLVER.SWA.LR_START = 2.5e-6
    _C.SOLVER.SWA.ETA_MIN_LR = 3.5e-6
    _C.SOLVER.SWA.LR_SCHED = False

    _C.SOLVER.REFERENCE_WORLD_SIZE = 0

    # input config
    _C.INPUT.FORMAT = "RGB"
    _C.INPUT.RESIZE_TYPE = "ResizeShortestEdge"


def add_efficientnet_config(cfg):

    _C = cfg
    _C.MODEL.EFFICIENTNET = CN()
    _C.MODEL.EFFICIENTNET.NAME = "efficientnet_b0"
    _C.MODEL.EFFICIENTNET.FEATURE_INDICES = [1, 4, 10, 15]
    _C.MODEL.EFFICIENTNET.OUT_FEATURES = [
        "stride4", "stride8", "stride16", "stride32"]


def add_hrnet_config(cfg):

    _C = cfg
    _C.MODEL.HRNET = CN()

    _C.MODEL.HRNET.OUT_FEATURES = ["stage1", "stage2", "stage3", "stage4"]

    # MODEL.HRNET related params
    _C.MODEL.HRNET.BASE_CHANNEL = [96, 96, 96, 96]
    _C.MODEL.HRNET.CHANNEL_GROWTH = 2
    _C.MODEL.HRNET.BLOCK_TYPE = "BottleneckWithFixedBatchNorm"
    _C.MODEL.HRNET.BRANCH_DEPTH = [3, 3, 3, 3]
    _C.MODEL.HRNET.NUM_BLOCKS = [6, 4, 4, 4]
    _C.MODEL.HRNET.NUM_LAYERS = [3, 3, 3]
    _C.MODEL.HRNET.FINAL_CONV_KERNEL = 1

    # for bi-directional fusion
    # Stage 1
    _C.MODEL.HRNET.STAGE1 = CN()
    _C.MODEL.HRNET.STAGE1.NUM_MODULES = 1
    _C.MODEL.HRNET.STAGE1.NUM_BRANCHES = 1
    _C.MODEL.HRNET.STAGE1.NUM_BLOCKS = [3]
    _C.MODEL.HRNET.STAGE1.NUM_CHANNELS = [64]
    _C.MODEL.HRNET.STAGE1.BLOCK = "BottleneckWithFixedBatchNorm"
    _C.MODEL.HRNET.STAGE1.FUSE_METHOD = "SUM"
    # Stage 2
    _C.MODEL.HRNET.STAGE2 = CN()
    _C.MODEL.HRNET.STAGE2.NUM_MODULES = 1
    _C.MODEL.HRNET.STAGE2.NUM_BRANCHES = 2
    _C.MODEL.HRNET.STAGE2.NUM_BLOCKS = [4, 4]
    _C.MODEL.HRNET.STAGE2.NUM_CHANNELS = [24, 48]
    _C.MODEL.HRNET.STAGE2.BLOCK = "BottleneckWithFixedBatchNorm"
    _C.MODEL.HRNET.STAGE2.FUSE_METHOD = "SUM"
    # Stage 3
    _C.MODEL.HRNET.STAGE3 = CN()
    _C.MODEL.HRNET.STAGE3.NUM_MODULES = 1
    _C.MODEL.HRNET.STAGE3.NUM_BRANCHES = 3
    _C.MODEL.HRNET.STAGE3.NUM_BLOCKS = [4, 4, 4]
    _C.MODEL.HRNET.STAGE3.NUM_CHANNELS = [24, 48, 92]
    _C.MODEL.HRNET.STAGE3.BLOCK = "BottleneckWithFixedBatchNorm"
    _C.MODEL.HRNET.STAGE3.FUSE_METHOD = "SUM"
    # Stage 4
    _C.MODEL.HRNET.STAGE4 = CN()
    _C.MODEL.HRNET.STAGE4.NUM_MODULES = 1
    _C.MODEL.HRNET.STAGE4.NUM_BRANCHES = 4
    _C.MODEL.HRNET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
    _C.MODEL.HRNET.STAGE4.NUM_CHANNELS = [24, 48, 92, 192]
    _C.MODEL.HRNET.STAGE4.BLOCK = "BottleneckWithFixedBatchNorm"
    _C.MODEL.HRNET.STAGE4.FUSE_METHOD = "SUM"
    _C.MODEL.HRNET.STAGE4.MULTI_OUTPUT = True
    # Decoder
    _C.MODEL.HRNET.DECODER = CN()
    _C.MODEL.HRNET.DECODER.BLOCK = "BottleneckWithFixedBatchNorm"
    _C.MODEL.HRNET.DECODER.HEAD_UPSAMPLING = "BILINEAR"
    _C.MODEL.HRNET.DECODER.HEAD_UPSAMPLING_KERNEL = 1
