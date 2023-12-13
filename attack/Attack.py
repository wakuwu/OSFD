import os
import mmcv
import logging
import functools

from mmcv.parallel import scatter
from attack.utils.buffer import Buffer
from attack.utils.mmdet import init_detector, init_cfg, init_dataloader, single_gpu_test, load_noise_for_eval
from attack.utils.pipelines import PreProcessData, CalculateLoss, UpdateNoise, PostProcessData
from attack.utils.registry import BASEATK, TSFATK
# keep this import
import attack.base, attack.comparing, attack.ours, ummdet.detectors, ummdet.components


class Attack:
    def __init__(self, yml_cfg) -> None:
        self.yml_cfg = yml_cfg
        self.model_cfg = init_cfg(yml_cfg.source_model_cfg[0], yml_cfg.dataset_cfg)
        self.model = init_detector(self.model_cfg,
                                   yml_cfg.source_model_cfg[1])
        self.dataloader = init_dataloader(self.model_cfg,
                                          samples_per_gpu=yml_cfg.dataloader_cfg.get("batch_size", 1),
                                          workers_per_gpu=yml_cfg.dataloader_cfg.get("cpu_num", 0),
                                          persistent_workers=yml_cfg.dataloader_cfg.get("persistent_workers", False))
        # Init attack methods
        self.base_attack = self.init_base_attack()
        self.transfer_attack = self.init_transfer_attack()

        # Init attack pipelines
        self.buffer = Buffer(os.environ.get("tmp_dir"))
        self.buffer.update_buffer_types(yml_cfg.buffer)
        self.pipeline = self.init_attack_pipeline()

        # Init Hooks
        self.attack_step_hooks = []

    def init_base_attack(self):
        base_attack = dict()
        for method in self.yml_cfg.attack_base:
            base_attack[method] = BASEATK.build(self.yml_cfg.default_cfg.base_attack[method])
        return base_attack

    def init_transfer_attack(self):
        transfer_attack = dict()
        method = self.yml_cfg.attack_transfer
        transfer_attack[method] = TSFATK.build(self.yml_cfg.default_cfg.transfer_attack[method])
        return transfer_attack

    def init_attack_pipeline(self):
        pre_process_data_pipeline = PreProcessData(self.base_attack, self.transfer_attack)
        calculate_loss_pipeline = CalculateLoss(self.base_attack, self.transfer_attack)
        update_noise_pipeline = UpdateNoise(self.base_attack)
        post_process_data_pipeline = PostProcessData(self.base_attack)
        return [pre_process_data_pipeline, calculate_loss_pipeline, update_noise_pipeline, post_process_data_pipeline]

    def attack_step(self, results):
        for p in self.pipeline[:-1]:
            results = p(results)
        for hook in self.attack_step_hooks:
            results = hook(results)
        return results

    def attack_epoch(self):
        mmcv.print_log("Generating adversarial examples.", logger="verbose_logger")
        losses = [0. for _ in range(self.yml_cfg.attack_cfg["steps"])]
        for idx, data in enumerate(mmcv.track_iter_progress(self.dataloader)):
            if "cuda" in self.yml_cfg.device:
                data = scatter(data, [0])[0]
            results = dict(idx=str(idx), data=data, buffer=self.buffer, model=self.model,
                           epsilon=self.yml_cfg.attack_cfg["epsilon"])
            for step in range(self.yml_cfg.attack_cfg["steps"]):
                results["step"] = step
                results = self.attack_step(results)
                # log the loss
                losses[step] += results.pop("loss_combined").item()
            self.pipeline[-1](results)
        return losses

    def eval(self, mode="clean"):
        metric_dict = dict()
        for model_name, (model_cfg_fp, model_checkpoint_fp) in self.yml_cfg.models_zoo.items():
            config = init_cfg(model_cfg_fp, self.yml_cfg.dataset_cfg)
            model = init_detector(config, model_checkpoint_fp)
            dataloader = init_dataloader(config,
                                         samples_per_gpu=self.yml_cfg.dataloader_cfg.get("eval_batch_size", 1),
                                         workers_per_gpu=self.yml_cfg.dataloader_cfg.get("eval_cpu_num", 0),
                                         persistent_workers=self.yml_cfg.dataloader_cfg.get("persistent_workers", False))
            if not "clean" == mode:
                func_load_noise = functools.partial(load_noise_for_eval, buffer=self.buffer,
                                                    buffer_batch_size=self.yml_cfg.dataloader_cfg["batch_size"])
            else:
                func_load_noise = None
            results = single_gpu_test(model, dataloader, func_load_noise, mode)

            eval_kwargs = config.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule', 'dynamic_intervals']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(**self.yml_cfg.eval_cfg))
            metric = dataloader.dataset.evaluate(results, logger=mmcv.get_logger("eval_logger"),
                                                 **eval_kwargs)
            metric_dict[model_name] = metric
            mmcv.print_log(f"The metric of {model_name} is: {metric}",
                           logger=mmcv.get_logger("eval_logger"), level=logging.INFO)
        return metric_dict