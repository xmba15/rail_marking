#!/usr/bin/env python
import os
import sys
import cv2
import torch
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import albumentations as abm

CURRENT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".")
sys.path.append(os.path.join(CURRENT_DIR, "../../"))
from rail_marking.segmentation.models import BiSeNetV2, OHEMCELoss  # noqa: E402
from rail_marking.segmentation.data_loader import EgoRailDataset, DataTransformBase  # noqa: E402
from rail_marking.segmentation.trainer import BiSeNetV2Trainer  # noqa: E402
from cfg import BiSeNetV2EgoConfig  # noqa: E402


def train_process(data_path, config):
    def _worker_init_fn_():
        import random
        import numpy as np
        import torch

        random_seed = config.random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)

    input_size = (config.img_height, config.img_width)

    PAD_VALUE = (0, 0, 0)
    IGNORE_INDEX = 255
    transforms = [
        abm.RandomResizedCrop(
            scale=(0.7, 1),
            ratio=(1.5, 2),
            height=config.img_height,
            width=config.img_width,
            interpolation=cv2.INTER_NEAREST,
            always_apply=True,
        ),
        abm.OneOf([abm.IAAAdditiveGaussianNoise(), abm.GaussNoise()], p=0.5),
        abm.OneOf(
            [
                abm.MedianBlur(blur_limit=3),
                abm.GaussianBlur(blur_limit=3),
                abm.MotionBlur(blur_limit=3),
            ],
            p=0.5,
        ),
        abm.OneOf(
            [
                abm.ShiftScaleRotate(
                    rotate_limit=7,
                    interpolation=cv2.INTER_NEAREST,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=PAD_VALUE,
                    mask_value=IGNORE_INDEX,
                    p=1.0,
                ),
                abm.ElasticTransform(
                    interpolation=cv2.INTER_NEAREST,
                    border_mode=cv2.BORDER_CONSTANT,
                    alpha_affine=30,
                    value=PAD_VALUE,
                    mask_value=IGNORE_INDEX,
                    p=1.0,
                ),
                abm.Perspective(
                    scale=(0.05),
                    interpolation=cv2.INTER_NEAREST,
                    pad_mode=cv2.BORDER_CONSTANT,
                    pad_val=PAD_VALUE,
                    mask_pad_val=IGNORE_INDEX,
                    keep_size=True,
                    fit_output=True,
                    p=1.0,
                ),
            ]
        ),
        abm.RandomGamma(gamma_limit=(80, 120), p=0.5),
        abm.RandomBrightnessContrast(brightness_limit=(-0.5, 0.5), contrast_limit=(-0.5, 0.5), p=0.5),
        abm.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        abm.RandomShadow(p=0.5),
        abm.ChannelShuffle(p=0.5),
        abm.ChannelDropout(p=0.5),
        abm.HorizontalFlip(p=0.5),
        abm.ImageCompression(quality_lower=50, p=0.5),
        abm.Cutout(num_holes=100, max_w_size=8, max_h_size=8, p=0.5),
    ]

    data_transform = DataTransformBase(transforms=transforms, input_size=input_size, normalize=True)
    train_dataset = EgoRailDataset(data_path=data_path, phase="train", transform=data_transform)
    val_dataset = EgoRailDataset(data_path=data_path, phase="val", transform=data_transform)

    # train_dataset.weighted_class()
    weighted_values = [8.90560578, 1.53155476]

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
        worker_init_fn=_worker_init_fn_(),
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=True,
    )
    data_loaders_dict = {"train": train_data_loader, "val": val_data_loader}
    model = BiSeNetV2(n_classes=config.num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = OHEMCELoss(thresh=config.ohem_ce_loss_thresh, weighted_values=weighted_values)

    base_lr_rate = config.lr_rate / (config.batch_size * config.batch_multiplier)
    base_weight_decay = config.weight_decay * (config.batch_size * config.batch_multiplier)

    def _lambda_epoch(epoch):
        import math

        max_epoch = config.num_epochs
        return math.pow((1 - epoch * 1.0 / max_epoch), 0.9)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=base_lr_rate,
        momentum=config.momentum,
        weight_decay=base_weight_decay,
    )
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=_lambda_epoch)
    trainer = BiSeNetV2Trainer(
        model=model,
        criterion=criterion,
        metric_func=None,
        optimizer=optimizer,
        data_loaders_dict=data_loaders_dict,
        config=config,
        scheduler=scheduler,
        device=device,
    )

    if config.snapshot and os.path.isfile(config.snapshot):
        trainer.resume_checkpoint(config.snapshot)

    with torch.autograd.set_detect_anomaly(True):
        trainer.train()


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--saved_model_path", type=str, required=True)
    parser.add_argument("--snapshot", type=str, required=False)
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    config = BiSeNetV2EgoConfig()
    config.saved_model_path = args.saved_model_path
    config.snapshot = args.snapshot
    train_process(args.data_path, config)


if __name__ == "__main__":
    main()
