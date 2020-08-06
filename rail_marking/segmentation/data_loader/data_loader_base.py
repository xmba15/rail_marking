#!/usr/bin/env python
import tqdm
import os
import numpy as np
import cv2
import torch.utils.data as data


class BaseDataset(data.Dataset):
    def __init__(self, data_path, classes, colors, phase, transform):
        super(BaseDataset, self).__init__()

        assert os.path.isdir(data_path)
        self._data_path = data_path

        self._image_paths = []
        self._gt_paths = []

        self._classes = classes
        self._colors = colors
        self._legend = BaseDataset.show_color_chart(self._classes, self._colors)

        assert phase in ("train", "val", "test")
        self._phase = phase

        self._transform = transform

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        image, gt = self._pull_item(idx)

        return image, gt

    def _pull_item(self, idx):
        image = cv2.imread(self._image_paths[idx])
        gt = cv2.imread(self._gt_paths[idx], 0)

        if self._transform is not None:
            image, gt = self._transform(image, gt, self._phase)

        return image, gt

    @property
    def colors(self):
        return self._colors

    @property
    def legend(self):
        return self._legend

    @property
    def classes(self):
        return self._classes

    @property
    def num_classes(self):
        return len(self._classes)

    def get_overlay_image(self, idx=None, image=None, label=None, alpha=0.7):
        if image is None or label is None:
            assert idx is not None and idx < self.__len__()
            image, label = self.__getitem__(idx)

        mask = np.array(self._colors)[label]
        overlay = (((1 - alpha) * image) + (alpha * mask)).astype("uint8")

        return overlay

    def class_distribution(self):
        assert self.__len__() > 0
        class_dist_dict = dict((el, 0) for el in self._classes)
        class_idx_dict = BaseDataset.class_to_class_idx_dict(self._classes)

        for idx in tqdm.tqdm(range(self.__len__())):
            _, gt = self.__getitem__(idx)
            for class_name in self._classes:
                class_dist_dict[class_name] += np.count_nonzero(gt == class_idx_dict[class_name])

        return class_dist_dict

    def weighted_class(self):
        class_dist_dict = self.class_distribution()
        total_pixels = np.sum(list(class_dist_dict.values()))
        class_idx_dict = BaseDataset.class_to_class_idx_dict(self._classes)

        weighted = np.zeros(self.num_classes, dtype=np.float64)
        for key, value in class_dist_dict.items():
            weighted[class_idx_dict[key]] = 1 / np.log(value * 1.0 / total_pixels + 1.02)

        return weighted

    @staticmethod
    def show_color_chart(classes, colors):
        legend = np.zeros(((len(classes) * 25) + 25, 300, 3), dtype="uint8")
        for (i, (class_name, color)) in enumerate(zip(classes, colors)):
            color = [int(c) for c in color]
            cv2.putText(
                legend,
                class_name,
                (5, (i * 25) + 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            cv2.rectangle(legend, (100, (i * 25)), (300, (i * 25) + 25), tuple(color), -1)

        return legend

    @staticmethod
    def class_to_class_idx_dict(classes):
        class_idx_dict = {}

        for i, class_name in enumerate(classes):
            class_idx_dict[class_name] = i

        return class_idx_dict

    @staticmethod
    def color_to_color_idx_dict(colors):
        color_idx_dict = {}

        for i, color in enumerate(colors):
            color_idx_dict[color] = i

        return color_idx_dict

    @staticmethod
    def human_sort(s):
        """Sort list the way humans do"""
        import re

        pattern = r"([0-9]+)"
        return [int(c) if c.isdigit() else c.lower() for c in re.split(pattern, s)]
