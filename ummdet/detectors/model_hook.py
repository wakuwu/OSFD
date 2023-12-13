from mmdet.models import SingleStageDetector, TwoStageDetector
import os
import torch


class ModelHook(object):
    """
    For FIA/NAA/RPA Attacks.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward_bottom(self, backbone_features, img_metas, gt_bboxes, gt_labels, return_loss=False):
        batch_input_shape = tuple(img_metas[0]["img_shape"][0:2])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        losses = None
        if isinstance(self, SingleStageDetector):
            losses = self.forward_bottom_one_stage(backbone_features, img_metas, gt_bboxes, gt_labels)
        elif isinstance(self, TwoStageDetector):
            losses = self.forward_bottom_two_stage(backbone_features, img_metas, gt_bboxes, gt_labels)

        # Accumulate model losses
        loss_accumulate = torch.tensor(0.0, device=os.environ["device"])
        for loss_name, loss_data in losses.items():
            if 'loss' in loss_name:
                if isinstance(loss_data, torch.Tensor):
                    loss_accumulate = loss_accumulate + loss_data
                if isinstance(loss_data, list):
                    for loss_item in loss_data:
                        loss_accumulate = loss_accumulate + loss_item
        # Return
        if losses is not None and not return_loss:
            loss_accumulate.backward()
            return
        else:
            return loss_accumulate

    def forward_bottom_one_stage(self, x, img_metas, gt_bboxes, gt_labels):
        if self.with_neck:
            x = self.neck(x)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels)
        return losses

    def forward_bottom_two_stage(self, x, img_metas, gt_bboxes, gt_labels):
        if self.with_neck:
            x = self.neck(x)
        losses = dict()
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=None,
                gt_masks=None,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = None
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels)
        losses.update(roi_losses)
        return losses