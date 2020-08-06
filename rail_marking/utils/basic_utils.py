#!/usr/bin/env python


__all__ = ["human_sort", "get_all_files_with_format_from_path"]


def human_sort(s):
    """Sort list the way humans do"""
    import re

    pattern = r"([0-9]+)"
    return [int(c) if c.isdigit() else c.lower() for c in re.split(pattern, s)]


def get_all_files_with_format_from_path(dir_path, suffix_format, use_human_sort=True):
    import os

    all_files = [elem for elem in os.listdir(dir_path) if elem.endswith(suffix_format)]
    if use_human_sort:
        all_files.sort(key=human_sort)

    return all_files
