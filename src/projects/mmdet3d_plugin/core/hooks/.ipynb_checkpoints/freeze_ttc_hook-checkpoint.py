# Copyright (c) OpenMMLab. All rights reserved.
"""Freeze all parameters except the TTC MLP (``*ttc_head*``)."""
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class FreezeDetectorExceptTTCHook(Hook):
    def __init__(self, substring="ttc_head"):
        self.substring = substring

    def before_run(self, runner):
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        for name, p in model.named_parameters():
            p.requires_grad = self.substring in name
