#!/usr/bin/env python
import numpy as np
import albumentations as abm
from albumentations.pytorch import ToTensor


__all__ = ["DataTransformBase"]


class DataTransformBase(object):
    def __init__(self, transforms, input_size, normalize=False):

        self._input_size = input_size
        self._normalize = normalize

        self._train_transform_list = []
        self._val_transform_list = []
        self._transforms_dict = {}

        self._transform_dict = {"train": {}, "val": {}}
        self._transform_dict["train"]["normal"] = transforms
        self._transform_dict["val"]["normal"] = []
        self._initialize_transform_dict()

    def _get_all_transforms_of_phase(self, phase):
        assert phase in ("train", "val")
        cur_transform = []
        cur_transform.extend(self._transform_dict[phase]["normal"])
        cur_transform.append(self._transform_dict[phase]["resize"])
        cur_transform.append(self._transform_dict[phase]["normalize"])

        return cur_transform

    def _initialize_transform_dict(self):
        height, width = self._input_size
        self._transform_dict["train"]["resize"] = abm.Resize(height, width, always_apply=True)
        self._transform_dict["val"]["resize"] = abm.Resize(height, width, always_apply=True)

        if self._normalize:
            self._transform_dict["train"]["normalize"] = abm.Normalize(always_apply=True)
            self._transform_dict["val"]["normalize"] = abm.Normalize(always_apply=True)
        else:
            self._transform_dict["train"]["normalize"] = ToTensor()
            self._transform_dict["val"]["normalize"] = ToTensor()

        self._transform_dict["train"]["all"] = self._get_all_transforms_of_phase("train")
        self._transform_dict["val"]["all"] = self._get_all_transforms_of_phase("val")

    def __call__(self, image, mask, phase):
        assert phase in ("train", "val")
        assert mask is not None

        transformer = abm.Compose(self._transform_dict[phase]["all"])
        augmented = transformer(image=image, masks=[mask])

        transformed_image = augmented["image"]
        transformed_mask = augmented["masks"][0]

        if self._normalize:
            transformed_image = np.transpose(transformed_image, (2, 0, 1))

        return transformed_image, transformed_mask
