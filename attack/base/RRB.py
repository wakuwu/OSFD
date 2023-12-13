import random

from .IFGSM import IFGSM
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rotate
from attack.utils.registry import BASEATK


@BASEATK.register_module()
class RRB(IFGSM):
    def __init__(self, alpha=1.0, prob=1.0, theta=10., l_s=10, rho=1.0, s_max=1.1, sigma=4.0) -> None:
        super().__init__(alpha)
        self.prob = prob
        self.theta = theta
        self.l_s = l_s
        self.rho = rho
        self.s_max = s_max
        self.sigma = sigma

    def preprocess_data(self, results):
        super().preprocess_data(results)
        data_adv_imgs = results.get("data_adv_imgs")
        data_gt_bboxes = results['data_gt_bboxes']
        data_adv_imgs_list = []
        for i in range(2):
            input_diversity = [self.random_axis_rotation, self.adaptive_random_resizing][i % 2]
            if random.random() < self.prob:
                data_adv_imgs = input_diversity(data_adv_imgs, max_angle=self.theta,
                                                label_boxes=data_gt_bboxes, factor=self.rho,
                                                max_scale=self.s_max, max_pixel=self.l_s)
            data_adv_imgs_list.append(data_adv_imgs)
        data_adv_imgs = torch.cat(data_adv_imgs_list, dim=0)
        data_adv_imgs = self.gaussian_blur(data_adv_imgs, sigma=self.sigma)
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
    def random_axis_rotation(imgs, theta=10., l_s=10, label_boxes=None, **kwargs):
        device = imgs.device
        result_list = []
        for idx in range(imgs.shape[0]):
            input_tensor = imgs[idx].unsqueeze(0)

            # Select the rotation axis randomly
            boxes = label_boxes[idx % len(label_boxes)]
            boxes_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            centers = torch.cat([boxes_centers, torch.tensor([[input_tensor.shape[-2] // 2,
                                                              input_tensor.shape[-1] // 2]], device=device)], dim=0)
            if l_s == 0:
                centers_with_random = centers
            else:
                centers_with_random = centers + torch.randint_like(centers, low=-l_s, high=l_s)
            center_x, center_y = random.choice(centers_with_random)
            angle = random.random() * 2 * theta - theta
            result = rotate(input_tensor, angle, center=[int(center_x), int(center_y)])
            result_list.append(result)
        return torch.cat(result_list, dim=0)

    @staticmethod
    def adaptive_random_resizing(imgs, rho=1.0, s_max=1.1, label_boxes=None, **kwargs):
        padded_list = []
        for idx in range(imgs.shape[0]):
            input_tensor = imgs[idx].unsqueeze(0)
            ori_size_h = input_tensor.shape[2]
            ori_size_w = input_tensor.shape[3]

            # Extract info of boxes
            boxes = label_boxes[idx % len(label_boxes)].cpu().numpy()
            random_box_idx = random.randint(0, len(boxes) - 1)
            box = boxes[random_box_idx]
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]

            # Apply Transformations
            scale_h = min(1 + rho * (box_h / ori_size_h), s_max)
            scale_w = min(1 + rho * (box_w / ori_size_w), s_max)
            new_size_h = random.randint(ori_size_h, int(scale_h * ori_size_h))
            new_size_w = random.randint(ori_size_w, int(scale_w * ori_size_w))
            rescaled = F.interpolate(input_tensor, size=(new_size_h, new_size_w), mode='bilinear', align_corners=True)
            rem_h = int(scale_h * ori_size_h) - new_size_h
            rem_w = int(scale_w * ori_size_w) - new_size_w
            pad_left = random.randint(0, rem_w)
            pad_top = random.randint(0, rem_h)
            padded = F.pad(rescaled, (pad_left, rem_w - pad_left, pad_top, rem_h - pad_top), mode='constant', value=0.)
            padded = F.interpolate(padded, size=(ori_size_h, ori_size_w), mode='bilinear', align_corners=True)
            padded_list.append(padded)
        return torch.cat(padded_list, dim=0)

    @staticmethod
    def gaussian_blur(imgs, sigma=1.0, **kwargs):
        return torch.clamp(imgs + torch.randn_like(imgs) * sigma, 0., 255.)