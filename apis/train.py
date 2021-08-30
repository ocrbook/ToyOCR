# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.
The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""
import detectron2
import logging
import time
import os
from detectron2.data import build_detection_test_loader

from detectron2.engine.defaults import DefaultTrainer
from detectron2.utils import comm
from detectron2.checkpoint import DetectionCheckpointer
from torch.nn.parallel import DistributedDataParallel
from detectron2.config import get_cfg
from configs import add_textnet_config

from modeling import * 
from detectron2.modeling import build_model


from data import DatasetMapper, build_detection_train_loader
from torchtools.optim import RangerLars
from solver import WarmupCosineAnnealingLR
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.engine import hooks
from . import additional_hooks

import torch


__all__ = ["Trainer"]


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        logger = logging.getLogger("detectron2")
        # setup_logger is not called for d2
        if not logger.isEnabledFor(logging.INFO):
            logger = setup_logger()
        cfg = Trainer.auto_scale_workers(cfg, comm.get_world_size())
        # Assume these objects must be constructed in this order.

        data_loader = self.build_train_loader(cfg)
        cfg = self.auto_scale_hyperparams(cfg, data_loader)

        model = self.build_model(cfg)

        optimizer = self.build_optimizer(cfg, model)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,find_unused_parameters=True
            )
        super(DefaultTrainer, self).__init__(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0

        if cfg.SOLVER.SWA.ENABLED:
            self.max_iter = cfg.SOLVER.MAX_ITER + cfg.SOLVER.SWA.ITER
        else:
            self.max_iter = cfg.SOLVER.MAX_ITER

        self.cfg = cfg
        self.skip_loss = cfg.MODEL.DETNET.LOSS.SKIP_LOSS
        self.history_loss = 10e8
        self.skip_weight = cfg.MODEL.DETNET.LOSS.SKIP_WEIGHT

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:
        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module in model.modules():
            for key, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                lr = cfg.SOLVER.BASE_LR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY
                if isinstance(module, norm_module_types):
                    weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
                elif key == "bias":
                    # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                    # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                    # hyperparameters are by default exactly the same as for regular
                    # weights.
                    lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                    weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
                params += [{"params": [value], "lr": lr,
                            "weight_decay": weight_decay}]

        assert cfg.SOLVER.OPTIM_NAME in ["RangerLars", "Adam", "SGD"]
        if cfg.SOLVER.OPTIM_NAME == "RangerLars":
            optimizer = RangerLars(params, lr=cfg.SOLVER.BASE_LR)
        if cfg.SOLVER.OPTIM_NAME == "Adam":
            optimizer = torch.optim.Adam(params, lr=cfg.SOLVER.BASE_LR)
        if cfg.SOLVER.OPTIM_NAME == "SGD":
            optimizer = torch.optim.SGD(
                params, lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, nesterov=cfg.SOLVER.NESTEROV
            )

        optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        if cfg.SOLVER.LR_SCHEDULER_NAME == "WarmupCosineAnnealingLR":
            decay_iter = int(cfg.SOLVER.MAX_ITER *
                             cfg.SOLVER.COSINE_DECAY_ITER)
            return WarmupCosineAnnealingLR(
                optimizer,
                max_iters=cfg.SOLVER.MAX_ITER,
                delay_iters=decay_iter,
                eta_min_lr=cfg.SOLVER.MIN_LR,
                warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                warmup_iters=cfg.SOLVER.WARMUP_ITERS,
                warmup_method=cfg.SOLVER.WARMUP_METHOD,
            )
        return build_lr_scheduler(cfg, optimizer)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        logger = logging.getLogger(__name__)
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
        ]

        if cfg.SOLVER.SWA.ENABLED:
            ret.append(
                additional_hooks.SWA(
                    cfg.SOLVER.MAX_ITER,
                    cfg.SOLVER.SWA.PERIOD,
                    cfg.SOLVER.SWA.LR_START,
                    cfg.SOLVER.SWA.ETA_MIN_LR,
                    cfg.SOLVER.SWA.LR_SCHED,
                )
            )

        if cfg.TEST.PRECISE_BN.ENABLED and hooks.get_bn_modules(self.model):
            logger.info("Prepare precise BN dataset")
            ret.append(hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            ))
        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(
                self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        return ret

    def run_step(self):
        """
        Implement the moco training logic described above.
        """
        assert self.model.training, "[KDTrainer] base model was changed to eval mode!"
<<<<<<< HEAD
=======
        
>>>>>>> 2b06fb0aea018e2d1515281a4593eb2f71867ffa

        
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
         
    
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        loss_dict = self.model(data)

        losses = sum(loss for loss in loss_dict.values())
        #self._detect_anomaly(losses, loss_dict)
        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        if self.skip_loss:
            if self.history_loss * self.skip_weight > losses.item():
                losses.backward()
                self.history_loss = losses.item()
            else:
                losses = 0.0 * losses
                losses.backward()
        else:
            losses.backward()
        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        metrics_dict = loss_dict
        if detectron2.__version__=="0.1.3":
            self._write_metrics(metrics_dict)
        elif detectron2.__version__<="0.2.1":
            self._write_metrics(metrics_dict,data_time)
        else:
            self._write_metrics(metrics_dict)

        self.optimizer.step()

    @staticmethod
    def auto_scale_hyperparams(cfg, data_loader):
        r"""
        This is used for auto-computation actual training iterations,
        because some hyper-param, such as MAX_ITER, means training epochs rather than iters,
        so we need to convert specific hyper-param to training iterations.
        """

        cfg = cfg.clone()
        frozen = cfg.is_frozen()
        cfg.defrost()

        iters_per_epoch = len(
            data_loader.dataset.dataset) // cfg.SOLVER.IMS_PER_BATCH
        cfg.SOLVER.MAX_ITER *= iters_per_epoch
        cfg.SOLVER.WARMUP_ITERS *= iters_per_epoch
        cfg.SOLVER.WARMUP_FACTOR = 1.0 / cfg.SOLVER.WARMUP_ITERS
        cfg.SOLVER.STEPS = list(cfg.SOLVER.STEPS)
        for i in range(len(cfg.SOLVER.STEPS)):
            cfg.SOLVER.STEPS[i] *= iters_per_epoch
        cfg.SOLVER.STEPS = tuple(cfg.SOLVER.STEPS)
        cfg.SOLVER.SWA.ITER *= iters_per_epoch
        cfg.SOLVER.SWA.PERIOD *= iters_per_epoch
        cfg.SOLVER.CHECKPOINT_PERIOD *= iters_per_epoch

        # Evaluation period must be divided by 200 for writing into tensorboard.
        num_mod = (200 - cfg.TEST.EVAL_PERIOD * iters_per_epoch) % 200
        cfg.TEST.EVAL_PERIOD = cfg.TEST.EVAL_PERIOD * iters_per_epoch + num_mod

        logger = logging.getLogger(__name__)
        logger.info(
            f"max_Iter={cfg.SOLVER.MAX_ITER}, wamrup_Iter={cfg.SOLVER.WARMUP_ITERS}, "
            f"step_Iter={cfg.SOLVER.STEPS}, ckpt_Iter={cfg.SOLVER.CHECKPOINT_PERIOD}, "
            f"eval_Iter={cfg.TEST.EVAL_PERIOD}."
        )

        if frozen:
            cfg.freeze()

        return cfg

    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.
        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        It is now implemented by:
        ::
            return [
                CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
            ]
        """
        # Here the default print/log frequency of each writer is used.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    @staticmethod
    def auto_scale_workers(cfg, num_workers: int):
        """
        When the config is defined for certain number of workers (according to
        ``cfg.SOLVER.REFERENCE_WORLD_SIZE``) that's different from the number of
        workers currently in use, returns a new cfg where the total batch size
        is scaled so that the per-GPU batch size stays the same as the
        original ``IMS_PER_BATCH // REFERENCE_WORLD_SIZE``.
        Other config options are also scaled accordingly:
        * training steps and warmup steps are scaled inverse proportionally.
        * learning rate are scaled proportionally, following :paper:`ImageNet in 1h`.
        For example, with the original config like the following:
        .. code-block:: yaml
            IMS_PER_BATCH: 16
            BASE_LR: 0.1
            REFERENCE_WORLD_SIZE: 8
            MAX_ITER: 5000
            STEPS: (4000,)
            CHECKPOINT_PERIOD: 1000
        When this config is used on 16 GPUs instead of the reference number 8,
        calling this method will return a new config with:
        .. code-block:: yaml
            IMS_PER_BATCH: 32
            BASE_LR: 0.2
            REFERENCE_WORLD_SIZE: 16
            MAX_ITER: 2500
            STEPS: (2000,)
            CHECKPOINT_PERIOD: 500
        Note that both the original config and this new config can be trained on 16 GPUs.
        It's up to user whether to enable this feature (by setting ``REFERENCE_WORLD_SIZE``).
        Returns:
            CfgNode: a new config. Same as original if ``cfg.SOLVER.REFERENCE_WORLD_SIZE==0``.
        """

        old_world_size =0# cfg.SOLVER.REFERENCE_WORLD_SIZE
        if old_world_size == 0 or old_world_size == num_workers:
            return cfg
        cfg = cfg.clone()
        frozen = cfg.is_frozen()
        cfg.defrost()

        assert (
            cfg.SOLVER.IMS_PER_BATCH % old_world_size == 0
        ), "Invalid REFERENCE_WORLD_SIZE in config!"
        scale = num_workers / old_world_size
        bs = cfg.SOLVER.IMS_PER_BATCH = int(round(cfg.SOLVER.IMS_PER_BATCH * scale))
        lr = cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * scale
        max_iter = cfg.SOLVER.MAX_ITER = int(round(cfg.SOLVER.MAX_ITER / scale))
        warmup_iter = cfg.SOLVER.WARMUP_ITERS = int(round(cfg.SOLVER.WARMUP_ITERS / scale))
        cfg.TEST.EVAL_PERIOD = int(round(cfg.TEST.EVAL_PERIOD / scale))
        cfg.SOLVER.CHECKPOINT_PERIOD = int(
            round(cfg.SOLVER.CHECKPOINT_PERIOD / scale))
        cfg.SOLVER.REFERENCE_WORLD_SIZE = num_workers  # maintain invariant
        logger = logging.getLogger(__name__)
        logger.info(
            f"Auto-scaling the config to batch_size={bs}, learning_rate={lr}, "
            f"max_iter={max_iter}, warmup={warmup_iter}."
        )

        if frozen:
            cfg.freeze()
        return cfg
