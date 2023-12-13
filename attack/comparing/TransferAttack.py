import os
import torch


class TransferAttack(object):

    def __init__(self) -> None:
        super().__init__()
        self.device = os.environ["device"]

    @staticmethod
    def extract_cln_features(results):
        model = results["model"]
        data_clean_imgs = results["data_clean_imgs"]
        normalizer = results["normalizer"]
        data_imgs = normalizer(data_clean_imgs)
        with torch.no_grad():
            feat_cln = model.backbone(data_imgs)
        return feat_cln

    def preprocess_data(self, results):
        buffer = results["buffer"]
        idx = results["idx"]
        # extract clean features
        feats_cln = results.get("feats_cln", None)
        if feats_cln is None:
            feats_cln = buffer.load_or_create_and_dump("feats_cln", idx,
                                                       function=self.extract_cln_features,
                                                       parameters=results)
        results["feats_cln"] = feats_cln
        return results

    def prepare_losses(self, results):
        loss_params = []
        feats_adv = results.pop("feats_adv")
        feats_cln = results["feats_cln"]
        # loss container
        num_stage = len(feats_cln)
        num_sample = results["num_sample"]
        num_groups = int(len(feats_adv[0]) / num_sample)
        for stage_idx in range(num_stage):
            for group_idx in range(num_groups):
                feats_adv_group = feats_adv[stage_idx][group_idx * num_sample: (group_idx + 1) * num_sample]
                for sample_idx in range(num_sample):
                    loss_params.append({
                        "group_idx": group_idx,
                        "stage_idx": stage_idx,
                        "sample_idx": sample_idx,
                        "feat_cln": feats_cln[stage_idx][sample_idx],
                        "feat_adv": feats_adv_group[sample_idx]
                    })
        results["loss_params"] = loss_params
        return results

    def losses(self, results):
        return results
