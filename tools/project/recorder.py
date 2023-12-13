import os
import copy
import numpy as np


class Recorder(object):

    def __init__(self, args) -> None:
        self.config = args
        # best params
        self.best_epoch = 0
        self.best_white_metric = None
        self.best_black_metric = None
        self.best_metric_dict = None
        # current
        self.metric_dict = None
        self.noise_dict = None
        self.white_metric = None
        self.black_metric = None
        self.loss_epoch = None
        self.loss_steps_one_epoch = None
        # saving to pkl
        self.loss_list = []
        # key:epoch value:eval_dict
        self.all_metric_dict = dict()
        os.environ["noise_dir"] = "noise_current"

    def _record_best(self, epoch):
        self.best_metric_dict = copy.deepcopy(self.metric_dict)
        self.best_epoch = epoch
        self.best_white_metric = self.white_metric
        self.best_black_metric = self.black_metric

    def _update_epoch(self, epoch, loss_step_list, metric_dict, samples_num, **kwargs):
        self.metric_dict = metric_dict
        # record loss
        loss_sum_epoch_ndarray = np.array(loss_step_list)
        loss_mean_epoch_ndarray = loss_sum_epoch_ndarray / samples_num
        self.loss_epoch = loss_mean_epoch_ndarray.mean()
        self.loss_steps_one_epoch = loss_mean_epoch_ndarray.tolist()
        self.loss_list.extend(self.loss_steps_one_epoch)
        # record metric
        self.white_metric, self.black_metric = self._calculate_metric(self.config.source_model_name, metric_dict)
        # record all metrics
        self.all_metric_dict[epoch] = metric_dict

    def record_clean(self, clean_metric_dict):
        self.all_metric_dict[0] = clean_metric_dict

    @staticmethod
    def _calculate_metric(source_name, metric_dict):
        tmp_dict = dict()
        white_metric = metric_dict.get(source_name, tmp_dict).get("bbox_mAP")
        black_metric = 0
        for name, metric in metric_dict.items():
            if name != source_name:
                black_metric += metric.get("bbox_mAP")
        black_metric /= (len(metric_dict) - 1)
        return white_metric, black_metric


class RecorderMemory(Recorder):

    def __init__(self, args) -> None:
        super(RecorderMemory, self).__init__(args)

    def record_best(self, epoch, buffer):
        noise_dict = buffer.buffer_dict["noise_current"]
        buffer.buffer_dict.pop("noise_best", None)
        buffer.buffer_dict["noise_best"] = copy.deepcopy(noise_dict)
        self._record_best(epoch)
        return

    def update_epoch(self, epoch, buffer, loss_step_list, metric_dict, samples_num, **kwargs):
        self._update_epoch(epoch, loss_step_list, metric_dict, samples_num, **kwargs)
        # Update the optimal solution
        if self.best_black_metric is None or self.black_metric < self.best_black_metric:
            self.record_best(epoch, buffer)


class RecorderDisk(Recorder):

    def __init__(self, args) -> None:
        super(RecorderDisk, self).__init__(args)

    def record_best(self, epoch):
        best_dir = os.environ.get("tmp_best_noise_buffer")
        current_dir = os.environ.get("tmp_current_noise_buffer")
        helper_dir = os.environ.get("tmp_helper")
        # deprecated best -> helper
        os.rename(best_dir, helper_dir)
        # current -> best
        os.rename(current_dir, best_dir)
        # helper -> current
        os.rename(helper_dir, current_dir)
        self._record_best(epoch)

    def update_epoch(self, epoch, loss_step_list, metric_dict, samples_num, **kwargs):
        self._update_epoch(epoch, loss_step_list, metric_dict, samples_num, **kwargs)
        # Update the optimal solution
        if self.best_black_metric is None or self.black_metric < self.best_black_metric:
            self.record_best(epoch)
            # make sure that the noise load for the next epoch is the best one
            os.environ["noise_dir"] = "noise_best"
        else:
            os.environ["noise_dir"] = "noise_current"
