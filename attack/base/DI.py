from .IFGSM import IFGSM
import torch
import random
import torch.nn.functional as F
from attack.utils.registry import BASEATK


@BASEATK.register_module()
class DI(IFGSM):
    def __init__(self, alpha=1.0, prob=0.7, scale=1.1) -> None:
        super().__init__(alpha)
        self.prob = prob
        self.scale = scale

    def preprocess_data(self, results):
        super().preprocess_data(results)
        data_adv_imgs = results.get("data_adv_imgs")
        if random.random() < self.prob:
            data_adv_imgs = self.input_diversity(data_adv_imgs, scale=self.scale)
        results["data_adv_imgs"] = data_adv_imgs
        return results

    def combine_losses(self, results):
        results = super().combine_losses(results)
        return results

    def process_gradients(self, results):
        results = super().process_gradients(results)
        return results

    def update_noise(self, results):
        results = super().update_noise(results)
        return results

    def postprocess_data(self, results):
        results = super().postprocess_data(results)
        return results

    @staticmethod
    def input_diversity(imgs, scale=1.1):
        padded_list = []
        for idx in range(imgs.shape[0]):
            input_tensor = imgs[idx].unsqueeze(0)
            ori_size = input_tensor.shape[2]
            new_size = random.randint(ori_size, int(scale * ori_size))
            rescaled = F.interpolate(input_tensor, size=(new_size, new_size), mode='bilinear', align_corners=True)
            rem = int(scale * ori_size) - new_size
            pad_left = random.randint(0, rem)
            pad_top = random.randint(0, rem)
            padded = F.pad(rescaled, (pad_left, rem - pad_left, pad_top, rem - pad_top), mode='constant', value=0.)
            padded = F.interpolate(padded, size=(ori_size, ori_size), mode='bilinear', align_corners=True)
            padded_list.append(padded)
        return torch.cat(padded_list, dim=0)