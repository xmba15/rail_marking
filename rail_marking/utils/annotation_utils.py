#!/usr/bin/env python
import cv2


def get_json_dict(json_path: str) -> dict:
    import json

    json_dict = None
    with open(json_path) as json_file:
        json_dict = json.load(json_file)
    return json_dict


def save_json_dict(output_json_path: str, json_dict: dict):
    import json

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)


def get_all_linetrips(json_object):
    label_shapes = json_object["shapes"]
    linestrips = []
    for shape in label_shapes:
        if shape["shape_type"] != "linestrip":
            continue
        linestrips.append(shape)
    return linestrips


def smoothen_linestrip(linestrip, downscale_length_ratio):
    from .path_smoothing import BezierCurve2D

    all_points = linestrip["points"]
    bezier_curve = BezierCurve2D(all_points, downscale_length_ratio=downscale_length_ratio)
    linestrip["points"] = bezier_curve.estimate_trajectory()


def smoothen_linestrips(linestrips, downscale_length_ratio):
    for i, _ in enumerate(linestrips):
        smoothen_linestrip(linestrips[i], downscale_length_ratio)


def smoothen_label(input_json_path, output_json_path, downscale_length_ratio):
    json_dict = get_json_dict(input_json_path)
    all_linestrips = get_all_linetrips(json_dict)
    smoothen_linestrips(all_linestrips, downscale_length_ratio)
    json_dict["shapes"] = all_linestrips
    save_json_dict(output_json_path, json_dict)


def get_all_pair_linestrips(all_linestrips):
    pair_num = (len(all_linestrips) + 1) // 2
    pair_linestrips = [None] * pair_num
    direction_map = {"left": 0, "right": 1}
    for linestrip in all_linestrips:
        linestrip_label = linestrip["label"]
        direction, number = linestrip_label.split("_")
        if not pair_linestrips[int(number) - 1]:
            pair_linestrips[int(number) - 1] = [None, None]
        pair_linestrips[int(number) - 1][direction_map[direction]] = linestrip

    for pair in pair_linestrips:
        left, right = pair
    return pair_linestrips


def visualize_linestrip(img, linestrip, smoothen, color, radius):
    if not linestrip:
        return

    if smoothen:
        smoothen_linestrip(linestrip)

    for (x, y) in linestrip["points"]:
        x = int(x)
        y = int(y)
        cv2.circle(img, (x, y), radius, color=color, thickness=-1)


def visualize_pair_linestrip(img, pair_linestrip, smoothen, color, radius):
    left_linestrip, right_linestrip = pair_linestrip
    visualize_linestrip(img, left_linestrip, smoothen, color, radius)
    visualize_linestrip(img, right_linestrip, smoothen, color, radius)


def visualize_all_pair_linestrips(img, all_pair_linestrips, smoothen=True, colors=None, radius=8):
    if not colors:
        from .color_utils import generate_color_chart

        colors = generate_color_chart(len(all_pair_linestrips))

    for pair_linestrip, color in zip(all_pair_linestrips, colors):
        visualize_pair_linestrip(img, pair_linestrip, smoothen, color, radius)


def get_image_label_lists(data_path: str, image_format=".jpg", label_format=".json"):
    from .basic_utils import get_all_files_with_format_from_path

    image_list = get_all_files_with_format_from_path(data_path, image_format)
    label_list = get_all_files_with_format_from_path(data_path, label_format)
    return image_list, label_list


def generate_smoothened_label(data_path, output_path, label_format, downscale_length_ratio=10):
    import os
    from .basic_utils import get_all_files_with_format_from_path

    label_list = get_all_files_with_format_from_path(data_path, label_format)
    for label in label_list:
        input_json_path = os.path.join(data_path, label)
        output_json_path = os.path.join(output_path, label)
        smoothen_label(input_json_path, output_json_path, downscale_length_ratio)
