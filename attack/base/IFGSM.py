from .BaseAttack import BaseAttack
from attack.utils.registry import BASEATK
import torch


@BASEATK.register_module()
class IFGSM(BaseAttack):
    def __init__(self, alpha=1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def preprocess_data(self, results):
        if results.get("data_adv_imgs") is None:
            data_adv_imgs = results["data_clean_imgs"] + results["noise"]
            results["data_adv_imgs"] = data_adv_imgs
        return results

    def combine_losses(self, results):
        # Sum the loss
        losses = results.pop("losses", None)
        if losses is not None:
            loss = torch.cat([l.get("loss_item")[None] for l in losses]).sum()
            results["loss_combined"] = loss
        return results

    def process_gradients(self, results):
        results = super().process_gradients(results)
        return results

    def update_noise(self, results):
        data_adv_imgs = results.pop("data_adv_imgs", None)
        if data_adv_imgs is not None:
            epsilon = results["epsilon"]
            gradients_adv = results.pop("gradients_adv", None)
            noise = results.pop("noise", None)
            noise = noise + self.alpha * torch.sign(gradients_adv)
            noise = torch.clamp(noise, min=-epsilon, max=epsilon)
            results["noise"] = noise
        return results

    def postprocess_data(self, results):
        results = super().postprocess_data(results)
        return results

