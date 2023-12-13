from attack.comparing.TransferAttack import TransferAttack
from attack.utils.registry import TSFATK

import torch.nn.functional as F


@TSFATK.register_module()
class OSFD(TransferAttack):
    def __init__(self, k=3.0) -> None:
        TransferAttack.__init__(self)
        self.k = k

    def preprocess_data(self, results):
        results = super().preprocess_data(results)
        return results

    def prepare_losses(self, results):
        results = super().prepare_losses(results)
        return results

    def losses(self, results):
        loss_params = results.pop("loss_params")
        for param in loss_params:
            feat_cln = param.pop("feat_cln")
            feat_adv = param.pop("feat_adv")
            l = F.mse_loss(self.k * feat_cln, feat_adv)
            param["loss_item"] = l
        results["losses"] = loss_params
        return results