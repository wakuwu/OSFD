import os
import yaml
import torch
import codecs
import random
import numpy as np
from tools.utils import get_utc8_time


class BaseConfig:
    def __init__(self, path: str) -> None:
        if not path:
            raise ValueError('Please specify the configuration file path.')
        if not os.path.exists(path):
            raise FileNotFoundError('File {} does not exist'.format(path))
        if path.endswith('yml') or path.endswith('yaml'):
            self._dic = self._parse_from_yaml(path)
        else:
            raise RuntimeError('Config file should in yaml format!')
        self._timestamp = get_utc8_time()
        self._setting_env()
        self._setting_seed()

    def _setting_env(self):
        os.environ["timestamp"] = self._timestamp
        os.environ["project_path"] = self._dic["global"]["project_path"]

    def _setting_seed(self):
        seed = self._dic["global"]["seed"]
        if seed is None:
            seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def _parse_from_yaml(path: str, full=True):
        """Parse a yaml file and build config"""
        with codecs.open(path, 'r', 'utf-8') as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)
        if full and 'base' in dic:
            project_dir = dic["global"]["project_path"]
            base_path = dic.pop('base')
            base_path = os.path.join(project_dir, base_path)
            base_dic = BaseConfig._parse_from_yaml(base_path)
            dic = BaseConfig._update_dic(dic, base_dic)
        return dic

    @staticmethod
    def _update_dic(dic, base_dic):
        """Update config from dic based base_dic"""
        base_dic = base_dic.copy()
        for key, val in dic.items():
            if isinstance(val, dict) and key in base_dic and base_dic.get(key) is not None:
                base_dic[key] = BaseConfig._update_dic(val, base_dic[key])
            else:
                base_dic[key] = val
        dic = base_dic
        return dic

    def __str__(self):
        return yaml.dump(self._dic)
