#!/usr/bin/env python
import os
import glob
import random
from .data_loader_base import BaseDataset


__all__ = ["EgoRailDatasetConfig", "EgoRailDataset"]


class EgoRailDatasetConfig:
    EGO_RAIL_CLASSES = [
        "ego-rail",
        "unidentified",
    ]

    EGO_RAIL_COLORS = [
        (0, 0, 255),
        (0, 0, 0),
    ]

    @property
    def num_classes(self):
        return len(self.EGO_RAIL_CLASSES)


class EgoRailDataset(BaseDataset):
    def __init__(
        self,
        data_path,
        phase="train",
        transform=None,
        random_seed=2020,
        train_val_ratio=0.9,
    ):
        super(EgoRailDataset, self).__init__(
            data_path,
            phase=phase,
            classes=EgoRailDatasetConfig.EGO_RAIL_CLASSES,
            colors=EgoRailDatasetConfig.EGO_RAIL_COLORS,
            transform=transform,
        )

        self._data_path = data_path
        assert os.path.isdir(self._data_path)

        _all_image_paths = glob.glob(os.path.join(self._data_path, "*[!mask].png"))
        _all_image_paths.sort(key=BaseDataset.human_sort)

        _all_gt_paths = glob.glob(os.path.join(self._data_path, "*_mask.png"))
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
