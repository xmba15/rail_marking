#!/usr/bin/env python
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
try:
    sys.path.append(os.path.join(_CURRENT_DIR, ".."))
    from rail_marking.utils import *
except Exception as e:
    print(e)
    sys.exit(-1)


def _has_rail_label(objects):
    for obj in objects:
        if obj["label"] == "rail":
            return True
    return False


def _process_objects(objects, pair_only):
    shapes = []
    count_line = 0
    for obj in objects:
        if obj["label"] != "rail":
            continue
        if pair_only and "polyline" in obj.keys():
            return None
        if "polyline-pair" in obj.keys():
            right_points, left_points = obj["polyline-pair"]
            left_dict = {
                "label": "line" + str(count_line),
                "points": left_points,
                "group_id": None,
                "shape_type": "linestrip",
                "flags": {},
            }
            right_dict = {
                "label": "line" + str(count_line + 1),
                "points": right_points,
                "group_id": None,
                "shape_type": "linestrip",
                "flags": {},
            }
            shapes.append(left_dict)
            shapes.append(right_dict)
            count_line += 2
        if "polyline" in obj.keys():
            single_line = {
                "label": "line" + str(count_line),
                "points": obj["polyline"],
                "group_id": None,
                "shape_type": "linestrip",
                "flags": {},
            }
            count_line += 1
            shapes.append(single_line)

    return shapes


def main(args):
    from shutil import copyfile
    import tqdm

    image_path = os.path.join(args.input_data_path, "jpgs/rs19_val")
    label_path = os.path.join(args.input_data_path, "jsons/rs19_val")
    image_list = get_all_files_with_format_from_path(image_path, ".jpg")
    label_list = get_all_files_with_format_from_path(label_path, ".json")
    assert len(image_list) != 0 and len(label_list) == len(image_list)

    for image_name, label_name in tqdm.tqdm(zip(image_list, label_list)):
        json_dict = get_json_dict(os.path.join(label_path, label_name))
        json_objects = json_dict["objects"]
        if not _has_rail_label(json_objects):
            continue
        shapes = _process_objects(json_objects, args.pair_only)
        if not shapes:
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
        if len(output_json_dict["shapes"]) > args.max_num:
            continue

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
    parser.add_argument("--max_num", type=int, default=10)
    parser.add_argument("--pair_only", action="store_true")
    parsed_args = parser.parse_args()

    if not os.path.isdir(parsed_args.output_data_path):
        raise Exception("path {} does not exist\n".format(parsed_args.output_data_path))

    main(parsed_args)
