# Copyright (c) OpenMMLab. All rights reserved.
from .model_hook import ModelHook
from mmdet.models import SingleStageDetector, DETECTORS


@DETECTORS.register_module()
class VFNetAdv(SingleStageDetector, ModelHook):
    """Implementation of `VarifocalNet
    (VFNet).<https://arxiv.org/abs/2008.13367>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(VFNetAdv, self).__init__(backbone, neck, bbox_head, train_cfg,
                                       test_cfg, pretrained, init_cfg)
