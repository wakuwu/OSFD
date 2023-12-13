import os
import os.path as osp

import mmcv

from tools.project.base_config import BaseConfig


class ConfigYaml(BaseConfig):
    def __init__(self, path: str) -> None:
        super(ConfigYaml, self).__init__(path)
        os.environ["device"] = self.device

    @staticmethod
    def build(path):
        yaml_config = ConfigYaml(path)
        # Init Config
        _config_dict = dict()
        for attribute in yaml_config.__dir__():
            if "_" != attribute[0] and "build" != attribute:
                _config_dict[attribute] = getattr(yaml_config, attribute)
        return mmcv.Config(_config_dict)

    @property
    def options(self):
        return self._dic.get("options")

    @property
    def saving_settings(self):
        return self._dic.get("saving_settings")

    @property
    def saving_dir(self):
        return self.saving_settings.get("saving_dir")

    @property
    def debug_mode(self):
        return self._dic.get("options").get("debug")

    @property
    def project_path(self):
        project_path = self._dic.get("global").get("project_path")
        os.environ["project_path"] = project_path
        return project_path

    @property
    def device(self):
        device = self._dic.get("global").get("device")
        return device

    @property
    def buffer(self):
        buffer_type = self._dic.get("global").get("buffer")
        noise_type = buffer_type.get("noise", buffer_type["global"])
        buffer_type["noise_current"] = noise_type
        buffer_type["best_current"] = noise_type
        return buffer_type

    @property
    def attack_cfg(self):
        return self._dic.get("attack")

    @property
    def attack_base(self):
        return self.attack_cfg.get("method").get("base_attack")

    @property
    def attack_transfer(self):
        return self.attack_cfg.get("method").get("transfer_attack")

    @property
    def default_cfg(self):
        default_cfg_fp = osp.join(self.project_path, self.attack_cfg.get("default_cfg"))
        default_cfg = mmcv.Config.fromfile(default_cfg_fp)
        # update default_cfg
        method_settings = self.attack_cfg.get("method").get("method_settings")
        if method_settings is not None:
            for base_method in self.attack_base:
                if method_settings.get(base_method) is not None:
                    default_cfg.base_attack[base_method].update(method_settings.get(base_method))
            if method_settings.get(self.attack_transfer) is not None:
                default_cfg.transfer_attack[self.attack_transfer].update(method_settings.get(self.attack_transfer))
        return default_cfg

    @property
    def dataloader_cfg(self):
        return self.attack_cfg.get("dataloader")

    @property
    def dataset_cfg(self):
        return self.attack_cfg.get("dataset")

    @property
    def eval_cfg(self):
        return self.attack_cfg.get("eval_cfg")

    @property
    def source_model_name(self):
        return self.attack_cfg.get("source")

    @property
    def source_model_cfg(self):
        cfg_dir = osp.join(self.project_path, self._dic.get("models").get("train_cfg_dir"))
        ckpt_dir = osp.join(self.project_path, self._dic.get("models").get("models_dir"))
        for model in self._dic.get("models").get("detectors"):
            model_name = model.get("name")
            if model_name == self.source_model_name:
                ckpt = model.get("ckpt")
                model_cfg_fp = osp.join(cfg_dir, model_name + ".py")
                model_checkpoint_fp = osp.join(ckpt_dir, ckpt + ".pth")
                return model_cfg_fp, model_checkpoint_fp

    @property
    def models_zoo(self):
        model_dict = dict()
        cfg_dir = osp.join(self.project_path, self._dic.get("models").get("eval_cfg_dir"))
        ckpt_dir = osp.join(self.project_path, self._dic.get("models").get("models_dir"))
        for model in self._dic.get("models").get("detectors"):
            model_name = model.get("name")
            ckpt = model.get("ckpt")
            model_cfg_fp = osp.join(cfg_dir, model_name + ".py")
            model_checkpoint_fp = osp.join(ckpt_dir, ckpt + ".pth")
            model_dict[model_name] = (model_cfg_fp, model_checkpoint_fp)
        return model_dict
