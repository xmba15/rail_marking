#!/usr/bin/env python
import os
import sys
import numpy as np

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
try:
    sys.path.append(os.path.join(_CURRENT_DIR, ".."))
    from rail_marking.core import RS19_CLASSES, RS19_COLORS
    from rail_marking.utils import get_all_files_with_format_from_path
except Exception as e:
    print(e)
    sys.exit(-1)


_TRAM_INDEX = RS19_CLASSES.index("tram-track")
_RAIL_RAISED_INDEX = RS19_CLASSES.index("rail-raised")
_RAIL_TRACK_INDEX = RS19_CLASSES.index("rail-track")


def _has_tram_label(mask_gt):
    return np.count_nonzero(mask_gt == _TRAM_INDEX) > 0


def _process_mask_gt(mask_gt):
    result = np.ones((mask_gt.shape[0], mask_gt.shape[1])) * 2
    result[mask_gt == _RAIL_RAISED_INDEX] = 0
    result[mask_gt == _RAIL_TRACK_INDEX] = 1

    return result


def main(args):
    import cv2
    import tqdm
    from shutil import copyfile

    image_path = os.path.join(args.input_data_path, "jpgs/rs19_val")
    mask_label_path = os.path.join(args.input_data_path, "uint8/rs19_val")
    image_list = get_all_files_with_format_from_path(image_path, ".jpg")
    mask_label_list = get_all_files_with_format_from_path(mask_label_path, ".png")
    assert len(image_list) != 0 and len(mask_label_list) == len(image_list)

    for image_name, label_name in tqdm.tqdm(zip(image_list, mask_label_list)):
        mask_gt = cv2.imread(os.path.join(mask_label_path, label_name), 0)
        if _has_tram_label(mask_gt):
            continue

        copyfile(
            os.path.join(image_path, image_name),
            os.path.join(args.output_data_path, image_name),
        )
        cv2.imwrite(os.path.join(args.output_data_path, label_name), _process_mask_gt(mask_gt))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, required=True)
    parser.add_argument("--output_data_path", type=str, required=True)
    parsed_args = parser.parse_args()

    if not os.path.isdir(parsed_args.output_data_path):
        raise Exception("path {} does not exist\n".format(parsed_args.output_data_path))

    main(parsed_args)
