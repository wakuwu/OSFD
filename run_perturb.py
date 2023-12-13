import os
import sys
import mmcv
import torch
import logging
import argparse
import traceback

from attack.Attack import Attack
from tools.project.config import ConfigYaml
from tools.project.logger import Logger
from tools.project.recorder import RecorderMemory, RecorderDisk


def parse_args():
    parser = argparse.ArgumentParser(description="Generate perturb images.")
    parser.add_argument('config', default="config/attack_yolov3.yaml", help='attack config file path')
    return parser.parse_args()


def run_main():
    # load config
    yml_cfg = ConfigYaml.build(parse_args().config)
    # Init Logger
    logger = Logger(yml_cfg, parse_args().config)

    # Init Recorder
    if "disk" == yml_cfg.buffer.get("nosie", yml_cfg.buffer["global"]):
        recorder = RecorderDisk(yml_cfg)
    else:
        recorder = RecorderMemory(yml_cfg)

    # Init Attack
    attack = Attack(yml_cfg)

    # eval clean mAP
    if yml_cfg.options.get("eval_clean"):
        clean_metric_dict = attack.eval(mode="clean")
        recorder.record_clean(clean_metric_dict)
        logger.logging_clean_metric(recorder)

    # main loop
    try:
        for epoch in range(yml_cfg.attack_cfg.get("max_epoch", 100)):
            losses = attack.attack_epoch()
            metric_dict = attack.eval(mode="noise")
            recorder.update_epoch(epoch, buffer=attack.buffer,
                                  loss_step_list=losses, metric_dict=metric_dict,
                                  samples_num=len(attack.dataloader.dataset))
            logger.logging_epoch(epoch, recorder=recorder)
    except Exception as e:
        mmcv.print_log(f"##### PID {os.getpid()} exit: error. #####", logger=mmcv.get_logger("verbose_logger"), level=logging.INFO)
        mmcv.print_log(str(e.args), logger=mmcv.get_logger("error_logger"), level=logging.ERROR)
        mmcv.print_log(traceback.format_exc(), logger=mmcv.get_logger("error_logger"), level=logging.ERROR)
        save_exit(logger=logger, recorder=recorder, attack=attack, save_type="best_black", exit_flag=True, exit_code=-1)
    save_exit(logger=logger, recorder=recorder, attack=attack, save_type="best_black", exit_flag=True, exit_code=0)


def save_exit(logger, recorder, attack, save_type="best_black", exit_flag=False, exit_code=0):
    logger.saving_results(recorder=recorder, attack=attack, save_type=save_type)
    if exit_flag:
        logger.close_logger()
        torch.cuda.empty_cache()
        sys.exit(exit_code)


if __name__ == '__main__':
    run_main()

