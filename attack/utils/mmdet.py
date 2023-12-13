from pathlib import Path
import os
import os.path as osp
import functools as ft
import mmcv
import torch
from torchvision import transforms
from mmcv.parallel import scatter
from mmcv.runner import load_checkpoint
from mmdet.core import get_classes, encode_mask_results
from mmdet.models import build_detector
from mmdet.datasets import build_dataset, build_dataloader, replace_ImageToTensor


def init_cfg(config, dataset_cfg):
    if isinstance(config, (str, Path)):
        config = mmcv.Config.fromfile(config)
    if 'pretrained' in config.model:
        config.model.pretrained = None
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None
    # modify the dataset settings
    config.data.test.ann_file = osp.join(os.environ["project_path"], dataset_cfg["ann_file"])
    config.data.test.img_prefix = osp.join(os.environ["project_path"], dataset_cfg["img_prefix"])
    return config


def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file."""
    model = build_detector(config.model, train_cfg=config.get('train_cfg'), test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = get_classes('coco')
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def init_dataloader(cfg, samples_per_gpu=1, workers_per_gpu=2, persistent_workers=False):
    test_dataloader_default_args = dict(
        samples_per_gpu=samples_per_gpu, workers_per_gpu=workers_per_gpu,
        dist=False, shuffle=False, persistent_workers=persistent_workers)
    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    return data_loader


def parse_data(data):
    """parse the data container in mmdet"""
    data_imgs = data.get("img", [None])[0]
    data_img_metas = data.get("img_metas", [[None]])[0]
    data_gt_bboxes = data.get("gt_bboxes", [[None]])[0]
    data_gt_labels = data.get("gt_labels", [[None]])[0]
    return data_imgs, data_img_metas, data_gt_bboxes, data_gt_labels


def get_normalize_tools(data_img_meta):
    """clean_imgs always in rgb channel"""
    device = os.environ["device"]
    # denormalized
    img_norm_cfg = data_img_meta.get("img_norm_cfg")
    mean = img_norm_cfg.get("mean")
    std = img_norm_cfg.get("std")
    to_rgb = img_norm_cfg.get("to_rgb")
    normalizer = ft.partial(imnormalize, mean=torch.from_numpy(mean).to(device),
                            std=torch.from_numpy(std).to(device), to_rgb=to_rgb)
    denormalizer = ft.partial(imdenormalize, mean=torch.from_numpy(mean).to(device),
                              std=torch.from_numpy(std).to(device), to_rgb=to_rgb)
    return normalizer, denormalizer


def imnormalize(img, mean, std, to_rgb):
    """
    Normalize an image with mean and std.
    Must convert the img to bgr first.
    """
    imgs = img
    if not to_rgb:
        if img.ndim == 4:
            imgs = img[:, [2, 1, 0], ...]
        elif img.ndim == 3:
            imgs = img[[2, 1, 0], ...]
    imgs = torch.div(torch.sub(imgs, mean[..., None, None]), std[..., None, None])
    return imgs


def imdenormalize(img, mean, std, to_rgb):
    """Denormalize an image with mean and std."""
    imgs = torch.mul(img, std[..., None, None]) + mean[..., None, None]
    if not to_rgb:
        if imgs.ndim == 4:
            imgs = imgs[:, [2, 1, 0], ...]
        elif imgs.ndim == 3:
            imgs = imgs[[2, 1, 0], ...]
    return imgs


def load_noise_for_eval(buffer, buffer_batch_size, sample_idx_range):
    batch_idx_start = sample_idx_range[0] // buffer_batch_size
    batch_idx_end = sample_idx_range[1] // buffer_batch_size
    idx_bias = sample_idx_range[0] % buffer_batch_size
    idx_length = sample_idx_range[1] - sample_idx_range[0] + 1
    idxes = [str(i) for i in range(batch_idx_start, batch_idx_end + 1)]
    noises = buffer.load("noise_current", idxes)
    if isinstance(noises, list) or isinstance(noises, tuple):
        noises = torch.cat(noises, dim=0)
    noises = noises[idx_bias:idx_bias + idx_length]
    return noises


def single_gpu_test(model,
                    data_loader,
                    func_load_noise=None,
                    mode="clean"):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    normalizer, denormalizer, resizer = None, None, None
    eval_batch_size = data_loader.batch_size
    for i, data in enumerate(data_loader):
        # scatter data to gpus
        data = scatter(data, [0])[0]

        # Load Noise
        if not "clean" == mode and func_load_noise is not None:
            data_imgs, data_img_metas, _, _ = parse_data(data)
            if normalizer is None or denormalizer is None:
                normalizer, denormalizer = get_normalize_tools(data_img_metas[0])
            data_clean_imgs = denormalizer(data_imgs)
            if resizer is None:
                resizer = transforms.Resize(data_clean_imgs.shape[-2:], antialias=True)
            sample_idx_range = (i * eval_batch_size, i * eval_batch_size + len(data_imgs) - 1)
            noises = func_load_noise(sample_idx_range=sample_idx_range)
            noises_resized = resizer.forward(noises)
            # Quantization
            data_adv_imgs = torch.round(data_clean_imgs + noises_resized).float()
            data_adv_imgs = normalizer(torch.clamp(data_adv_imgs, min=0., max=255.))
            data["img"][0] = data_adv_imgs

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]

        batch_size = len(result)
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results
