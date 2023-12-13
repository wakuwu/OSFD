import os
import mmcv
import torch
import shutil
import logging
import traceback
import os.path as osp
import torchvision.utils as vutils
from mmcv.parallel import scatter
from attack.Attack import Attack
from attack.utils.mmdet import parse_data, get_normalize_tools
from tools.utils import get_file_name
from torch.utils.tensorboard import SummaryWriter


class Logger:

    all_base_methods = ["MI", "DI", "RRB", "IFGSM"]

    def __init__(self, yaml_cfg, yaml_cfg_path) -> None:
        self.attack_logger = None
        self.eval_logger = None
        self.verbose_logger = None
        self.tb_writer = None

        self.config = yaml_cfg
        self.saving_path_manager = dict()

        if not self.config.debug_mode or "disk" in self.config.buffer.values():
            self._init_saving_setting(yaml_cfg_path)

        # assist variables
        self.already_log_step_idx = 0
        self.already_log_clean = False

        # print timestamp
        mmcv.print_log("\nPID: " + str(os.getpid()), mmcv.get_logger("verbose_logger"))
        mmcv.print_log(os.environ.get("timestamp"), mmcv.get_logger("verbose_logger"))

    def _init_saving_setting(self, yaml_cfg_path):
        saving_settings = self.config.saving_settings
        project_path = self.config.project_path

        base_methods = self.config.attack_cfg.get("method").get("base_attack")
        base_method_name = []
        for m in self.all_base_methods:
            if m in base_methods:
                base_method_name.append(m)
        base_method_name = ''.join([x[0] for x in base_method_name])
        perturb_params = '_' + str(self.config.attack_cfg.get("epsilon")) + 'p'

        tsf_method = self.config.attack_cfg.get("method").get("transfer_attack")

        saving_dir = osp.join(project_path, "data/results",
                              self.config.saving_dir.get("prefix", ""),
                              self.config.source_model_name,
                              base_method_name + perturb_params,
                              tsf_method,
                              self.config.saving_dir.get("suffix", ""),
                              os.environ.get("timestamp"))
        mmcv.mkdir_or_exist(saving_dir)
        self.saving_path_manager["global_dir"] = saving_dir

        # buffer
        if "disk" in self.config.buffer.values():
            tmp_dir = osp.join(saving_dir, "tmp")
            mmcv.mkdir_or_exist(tmp_dir)
            os.environ["tmp_dir"] = tmp_dir
            if "disk" == self.config.buffer.get("noise", self.config.buffer["global"]):
                tmp_current_noise_buffer = osp.join(saving_dir, "tmp", "noise_current")
                tmp_best_noise_buffer = osp.join(saving_dir, "tmp", "noise_best")
                tmp_helper = osp.join(saving_dir, "tmp", "helper")

                mmcv.mkdir_or_exist(tmp_current_noise_buffer)
                mmcv.mkdir_or_exist(tmp_best_noise_buffer)

                os.environ["tmp_current_noise_buffer"] = tmp_current_noise_buffer
                os.environ["tmp_best_noise_buffer"] = tmp_best_noise_buffer
                os.environ["tmp_helper"] = tmp_helper

        if not self.config.debug_mode:
            # config file backup dir
            config_dir = osp.join(saving_dir, "config")
            mmcv.mkdir_or_exist(config_dir)
            self._config_backup(config_dir, yaml_cfg_path)

            # log file dir
            if saving_settings.get("logging"):
                log_dir = osp.join(saving_dir, "log")
                mmcv.mkdir_or_exist(log_dir)
                self._init_logger(log_dir)

            # tensorboard dir
            if saving_settings.get("tboard"):
                tensorboard_dir = osp.join(saving_dir, "tensorboard")
                mmcv.mkdir_or_exist(tensorboard_dir)
                self._init_tensorboard(tensorboard_dir)

            saving_best_white = saving_settings.get("best_white")
            saving_best_black = saving_settings.get("best_black")

            # noise dir
            if saving_settings.get("noise"):
                noise_dir = osp.join(saving_dir, "noise")
                self.saving_path_manager["noise_dir"] = noise_dir
                if saving_best_white:
                    best_white = osp.join(noise_dir, "best_white")
                    mmcv.mkdir_or_exist(best_white)
                if saving_best_black:
                    best_black = osp.join(noise_dir, "best_black")
                    mmcv.mkdir_or_exist(best_black)

            # adv imgs dir
            if saving_settings.get("adv_img"):
                adv_imgs_dir = osp.join(saving_dir, "adv_imgs")
                self.saving_path_manager["adv_imgs_dir"] = adv_imgs_dir
                if saving_best_white:
                    best_white = osp.join(adv_imgs_dir, "best_white")
                    mmcv.mkdir_or_exist(best_white)
                if saving_best_black:
                    best_black = osp.join(adv_imgs_dir, "best_black")
                    mmcv.mkdir_or_exist(best_black)

            # others
            if saving_settings.get("others"):
                others_dir = osp.join(saving_dir, "others")
                self.saving_path_manager["others_dir"] = others_dir
                if saving_best_white:
                    best_white = osp.join(others_dir, "best_white")
                    mmcv.mkdir_or_exist(best_white)
                if saving_best_black:
                    best_black = osp.join(others_dir, "best_black")
                    mmcv.mkdir_or_exist(best_black)
        return

    def _init_logger(self, log_dir):
        """Init log logger"""
        attack_logger_fp = osp.join(log_dir, "attack_log.txt")
        eval_logger_fp = osp.join(log_dir, "eval_log.txt")
        error_logger_fp = osp.join(log_dir, "error_log.txt")
        verbose_logger_fp = osp.join(log_dir, "verbose_log.txt")
        self.attack_logger = mmcv.get_logger("attack_logger", attack_logger_fp)
        self.eval_logger = mmcv.get_logger("eval_logger", eval_logger_fp)
        self.error_logger = mmcv.get_logger("error_logger", error_logger_fp)
        self.verbose_logger = mmcv.get_logger("verbose_logger", verbose_logger_fp)
        return

    def _config_backup(self, config_dir, yaml_cfg_path):
        """copy current configuration to results dir"""
        # backup base yaml config
        project_path = self.config.project_path
        shutil.copyfile(osp.join(project_path, "config/base.yaml"),
                        osp.join(config_dir, "base.yaml"))
        # backup default config
        shutil.copyfile(osp.join(project_path, "config/default.py"),
                        osp.join(config_dir, "default.py"))
        # backup attack yaml config
        shutil.copyfile(osp.join(project_path, yaml_cfg_path),
                        osp.join(config_dir, osp.split(yaml_cfg_path)[-1]))
        # backup train model config in mmcls
        shutil.copyfile(osp.join(project_path, "ummdet/checkpoints/train_cfg", self.config.source_model_name + ".py"),
                        osp.join(config_dir, self.config.source_model_name + ".py"))
        return

    def _init_tensorboard(self, log_dir):
        """Init TB"""
        self.tb_writer = SummaryWriter(log_dir=log_dir, flush_secs=60)

    def logging_clean_metric(self, recorder):
        if self.tb_writer is not None:
            """write clean metric to TB"""
            black_metric, white_metric = getattr(recorder, "_calculate_metric")(recorder.config.source_model_name,
                                                                                recorder.all_metric_dict[0])
            self.tb_writer.add_scalar("metric_white", white_metric, 0)
            self.tb_writer.add_scalar("metric_black", black_metric, 0)

    def logging_epoch(self, epoch, recorder):
        epoch = epoch + 1
        accumulate_steps = recorder.config.attack_cfg.get("steps") * (epoch - 1)
        # record loss in step
        for step, loss in enumerate(recorder.loss_steps_one_epoch):
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("loss_step", loss, accumulate_steps + step + 1)
            mmcv.print_log(f"step: {accumulate_steps + step + 1}   loss: {loss}", logger=self.attack_logger,
                           level=logging.INFO)

        # record loss in epoch
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("loss_epoch", recorder.loss_epoch, accumulate_steps + 1)
            # record metric
            self.tb_writer.add_scalar("metric_white", recorder.white_metric, accumulate_steps + 1)
            self.tb_writer.add_scalar("metric_black", recorder.black_metric, accumulate_steps + 1)

        mmcv.print_log(f"epoch: {epoch}   loss: {recorder.loss_epoch}\n"
                       f"metric_white: {recorder.white_metric}   metric_black: {recorder.black_metric}",
                       logger=self.attack_logger,
                       level=logging.INFO)

    def saving_results(self, recorder, attack: Attack, save_type="best_black"):
        """save all results from memory to disk"""
        # All results will be threw away in debug mode.
        if self.config.debug_mode:
            return
        saving_settings = self.config.saving_settings
        if not (saving_settings.get(save_type) is not None and saving_settings.get(save_type)):
            return

        # saving adversarial images and noise
        if saving_settings.get("adv_img") or saving_settings.get("noise"):
            dataloader = attack.dataloader
            noise_dir = "noise_current" if "best_white" == save_type else "noise_best"
            try:
                mmcv.print_log("Start saving adversarial images and noise ...",
                               logger=self.verbose_logger, level=logging.INFO)
                normalizer, denormalizer = None, None
                for batch_idx, data in enumerate(mmcv.track_iter_progress(dataloader)):
                    data = scatter(data, [0])[0]
                    data_imgs, data_img_metas, _, _ = parse_data(data)
                    data_noises = attack.buffer.load(noise_dir, str(batch_idx))
                    if saving_settings.get("adv_img"):
                        if normalizer is None or denormalizer is None:
                            normalizer, denormalizer = get_normalize_tools(data_img_metas[0])
                        data_clean_imgs = denormalizer(data_imgs)
                        # Quantization
                        data_adv_imgs = torch.clamp(torch.round(data_clean_imgs + data_noises), min=0., max=255.)

                    for idx, image_metas in enumerate(data_img_metas):
                        image_name = get_file_name(image_metas.get("ori_filename"), with_ext=False)
                        if saving_settings.get("noise"):
                            torch.save(data_noises[idx].detach().cpu(),
                                       osp.join(self.saving_path_manager.get("noise_dir"),
                                                save_type, image_name + ".pth"))
                        if saving_settings.get("adv_img"):
                            vutils.save_image(data_adv_imgs[idx].detach().cpu(),
                                              osp.join(self.saving_path_manager.get("adv_imgs_dir"),
                                                       f"{save_type}",
                                                       image_name + ".png"),
                                              normalize=True,
                                              value_range=(0, 255))

                mmcv.print_log("Saving saving adversarial images and noise done!", logger=self.verbose_logger,
                               level=logging.INFO)
            except Exception as e:
                mmcv.print_log(str(e.args), logger=self.error_logger,
                               level=logging.ERROR)
                mmcv.print_log(traceback.format_exc(), logger=self.error_logger,
                               level=logging.ERROR)

        # saving others to pkl
        if saving_settings.get("others"):
            try:
                mmcv.print_log("Start saving others...", logger=self.verbose_logger, level=logging.INFO)
                others_dir = self.saving_path_manager.get("others_dir")
                mmcv.dump(recorder.loss_list, osp.join(others_dir, f"{save_type}", "loss_step.pkl"))
                mmcv.dump(recorder.all_metric_dict, osp.join(others_dir, f"{save_type}", "all_metric_dict.pkl"))
                best_params = dict()
                best_params["best_epoch"] = recorder.best_epoch
                best_params["best_white_metric"] = recorder.best_white_metric
                best_params["best_black_metric"] = recorder.best_black_metric
                best_params["best_metric_dict"] = recorder.best_metric_dict
                mmcv.dump(best_params, osp.join(others_dir, f"{save_type}", "best_params.pkl"))
                mmcv.dump(best_params, osp.join(others_dir, f"{save_type}", "best_params.yaml"))
                mmcv.print_log("Saving others done!", logger=self.verbose_logger, level=logging.INFO)
            except Exception as e:
                mmcv.print_log(str(e.args), logger=self.error_logger,
                               level=logging.ERROR)
                mmcv.print_log(traceback.format_exc(), logger=self.error_logger,
                               level=logging.ERROR)
        return

    def close_logger(self):
        # Close Tb_writer
        if self.tb_writer is not None:
            self.tb_writer.close()
        # clear tmp dir
        if "disk" in self.config.buffer.values():
            shutil.rmtree(os.environ["tmp_dir"])
