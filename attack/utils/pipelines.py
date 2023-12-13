import os
import torch
from attack.utils.mmdet import parse_data, get_normalize_tools


class PreProcessData:
    def __init__(self, base_attack, transfer_attack) -> None:
        self.base_priority = ["DI", "RRB", "MI", "IFGSM"]
        self.device = os.environ["device"]
        self.base_attack = base_attack
        self.transfer_attack = transfer_attack

    @staticmethod
    def init_data_container(results):
        buffer = results["buffer"]
        data = results.pop("data", None)
        if data is None:
            return results
        data_imgs, data_img_metas, data_gt_bboxes, data_gt_labels = parse_data(data)
        normalizer, denormalizer = \
            buffer.load_or_create_and_dump_var("normalize_tools", get_normalize_tools, data_img_metas[0])
        data_clean_imgs = denormalizer(data_imgs)
        results["data_img_metas"] = data_img_metas
        results["data_gt_bboxes"] = data_gt_bboxes
        results["data_gt_labels"] = data_gt_labels
        results["normalizer"] = normalizer
        results["data_clean_imgs"] = data_clean_imgs
        results["num_sample"] = len(data_clean_imgs)
        return results

    def init_noise(self, results):
        buffer = results["buffer"]
        idx = results["idx"]
        noise = results.get("noise", None)
        if noise is None:
            noise_dir = os.environ["noise_dir"]
            noise = buffer.load(noise_dir, idx)
        if noise is None:
            noise = torch.randint_like(results["data_clean_imgs"], low=-2, high=3).float().to(self.device)
        noise.requires_grad = True
        # Init the gradients of noise
        results["noise"] = noise
        return results

    @staticmethod
    def init_attack_forward(results):
        model = results["model"]
        normalizer = results["normalizer"]
        data_adv_imgs = results["data_adv_imgs"]
        feats_adv = model.backbone(normalizer(torch.clamp(data_adv_imgs, min=0., max=255.)))
        results["feats_adv"] = feats_adv
        return results

    def __call__(self, results):
        results = self.init_data_container(results)
        # Init Noise
        results = self.init_noise(results)
        # Base Attack
        for method in self.base_priority:
            if method in self.base_attack.keys():
                method = self.base_attack[method]
                results = method.preprocess_data(results)
        # Tsf Attack
        for method in self.transfer_attack.keys():
            method = self.transfer_attack[method]
            results = method.preprocess_data(results)
        # Init features of adv images
        results = self.init_attack_forward(results)
        return results


class CalculateLoss:
    def __init__(self, base_attack, transfer_attack) -> None:
        self.base_priority = ["MI", "DI", "RRB", "IFGSM"]
        self.base_attack = base_attack
        self.transfer_attack = transfer_attack

    def __call__(self, results):
        # Tsf Attack
        for method in self.transfer_attack.keys():
            method = self.transfer_attack[method]
            results = method.prepare_losses(results)
            results = method.losses(results)
        # Combine losses
        for method in self.base_priority:
            if method in self.base_attack.keys():
                method = self.base_attack[method]
                results = method.combine_losses(results)
        # Backward to get gradients
        loss_combined = results["loss_combined"]
        loss_combined.backward()
        return results


class UpdateNoise:
    def __init__(self, base_attack) -> None:
        self.base_gradients_priority = ["MI", "DI", "RRB", "IFGSM"]
        self.base_update_priority = ["MI", "DI", "RRB", "IFGSM"]
        self.base_attack = base_attack

    @torch.no_grad()
    def __call__(self, results):
        with torch.no_grad():
            results["gradients_adv"] = results["noise"].grad
            # Handle gradients
            for method in self.base_gradients_priority:
                if method in self.base_attack.keys():
                    method = self.base_attack[method]
                    results = method.process_gradients(results)
        # Update noise
        for method in self.base_update_priority:
            if method in self.base_attack.keys():
                method = self.base_attack[method]
                results = method.update_noise(results)
        return results


class PostProcessData:
    def __init__(self, base_attack) -> None:
        super().__init__()
        self.base_attack = base_attack

    @staticmethod
    def buffer_noise(results):
        buffer = results["buffer"]
        idx = results["idx"]
        noise = results["noise"]
        buffer.dump("noise_current", idx, noise)
        return results

    def __call__(self, results):
        for method in self.base_attack.keys():
            method = self.base_attack[method]
            results = method.postprocess_data(results)
        results = self.buffer_noise(results)
        return results
