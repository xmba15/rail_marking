#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import cv2
import random
from .data_loader_base import BaseDataset


__all__ = ["Rs19DatasetConfig", "Rs19dDataset"]


class Rs19DatasetConfig:
    RS19_CLASSES = [
        "rail-raised",
        "rail-track",
        "unidentified",
    ]

    RS19_COLORS = [
        (0, 0, 255),
        (255, 255, 102),
        (0, 0, 0),
    ]

    @property
    def num_classes(self):
        return len(self.RS19_CLASSES)


class Rs19dDataset(BaseDataset):
    def __init__(
        self,
        data_path,
        phase="train",
        transform=None,
        random_seed=2020,
        train_val_ratio=0.9,
    ):
        super(Rs19dDataset, self).__init__(
            data_path,
            phase=phase,
            classes=Rs19DatasetConfig.RS19_CLASSES,
            colors=Rs19DatasetConfig.RS19_COLORS,
            transform=transform,
        )

        self._data_path = data_path
        assert os.path.isdir(self._data_path)

        _all_image_paths = glob.glob(os.path.join(self._data_path, "*.jpg"))
        _all_image_paths.sort(key=BaseDataset.human_sort)

        _all_gt_paths = glob.glob(os.path.join(self._data_path, "*.png"))
        _all_gt_paths.sort(key=BaseDataset.human_sort)

        zipped = list(zip(_all_image_paths, _all_gt_paths))
        random.seed(random_seed)
        random.shuffle(zipped)
        _all_image_paths, _all_gt_paths = zip(*zipped)

        _train_len = int(train_val_ratio * len(_all_image_paths))
        if self._phase == "train":
            self._image_paths = _all_image_paths[:_train_len]
            self._gt_paths = _all_gt_paths[:_train_len]
        else:
            self._image_paths = _all_image_paths[_train_len:]
            self._gt_paths = _all_gt_paths[_train_len:]

        self._color_idx_dict = BaseDataset.color_to_color_idx_dict(self._colors)

    def _pull_item(self, idx):
        image = cv2.imread(self._image_paths[idx])
        gt = cv2.imread(self._gt_paths[idx], 0)

        if self._transform is not None:
            image, gt = self._transform(image, gt, self._phase)

        return image, gt
