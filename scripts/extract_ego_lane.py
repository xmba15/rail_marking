#!/usr/bin/env python
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, ".."))
try:
    from rail_marking.utils import get_json_dict, get_all_files_with_format_from_path, save_json_dict
except Exception as e:
    print(e)
    sys.exit(-1)


def _has_rail_label(objects):
    for obj in objects:
        if obj["label"] == "rail":
            return True
    return False


def _points_distance(pt1, pt2):
    import math

    return math.sqrt(math.pow(pt1[0] - pt2[0], 2) + math.pow(pt1[1] - pt2[1], 2))


def _process_objects(objects, image_width, image_height):
    center_bottom = [image_width / 2, image_height - 1]

    left_points = None
    right_points = None
    cur_dist = float("inf")

    for obj in objects:
        if obj["label"] != "rail":
            continue
        if "polyline" in obj.keys():
            continue

        if "polyline-pair" in obj.keys():
            cur_right_points, cur_left_points = obj["polyline-pair"]
            if cur_right_points[0][0] < cur_left_points[0][0]:
                right_points, left_points = left_points, right_points
            sum_dist_to_center_bottom = _points_distance(cur_right_points[0], center_bottom) + _points_distance(
                cur_left_points[0], center_bottom
            )
            if sum_dist_to_center_bottom < cur_dist:
                cur_dist = sum_dist_to_center_bottom
                left_points = cur_left_points
                right_points = cur_right_points

    if left_points is None or right_points is None:
        return None

    shapes = []
    left_dict = {
        "label": "ego_left",
        "points": left_points,
        "group_id": None,
        "shape_type": "linestrip",
        "flags": {},
    }
    right_dict = {
        "label": "ego_right",
        "points": right_points,
        "group_id": None,
        "shape_type": "linestrip",
        "flags": {},
    }
    shapes.append(left_dict)
    shapes.append(right_dict)

    return shapes


def main(args):
    from shutil import copyfile
    import tqdm

    image_path = os.path.join(args.input_data_path, "jpgs/rs19_val")
    label_path = os.path.join(args.input_data_path, "jsons/rs19_val")
    image_list = get_all_files_with_format_from_path(image_path, ".jpg")
    label_list = get_all_files_with_format_from_path(label_path, ".json")
    assert len(image_list) != 0 and len(label_list) == len(image_list)

    for image_name, label_name in zip(tqdm.tqdm(image_list), label_list):
        json_dict = get_json_dict(os.path.join(label_path, label_name))
        json_objects = json_dict["objects"]
        if not _has_rail_label(json_objects):
            continue

        shapes = _process_objects(json_objects, json_dict["imgWidth"], json_dict["imgHeight"])
        if shapes is None:
            continue

        output_json_dict = {
            "version": "4.4.0",
            "imageData": None,
            "flags": {},
            "shapes": shapes,
            "imagePath": image_name,
            "imageHeight": json_dict["imgHeight"],
            "imageWidth": json_dict["imgWidth"],
        }
        save_json_dict(os.path.join(args.output_data_path, label_name), output_json_dict)
        copyfile(
            os.path.join(image_path, image_name),
            os.path.join(args.output_data_path, image_name),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, required=True)
    parser.add_argument("--output_data_path", type=str, required=True)
    parsed_args = parser.parse_args()

    if not os.path.isdir(parsed_args.output_data_path):
        raise Exception("path {} does not exist".format(parsed_args.output_data_path))

    main(parsed_args)
