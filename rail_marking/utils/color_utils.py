#!/usr/bin/env python


__all__ = ["generate_color_chart"]


def generate_color_chart(num_colors, seed=3700):
    import numpy as np

    assert num_colors > 0
    np.random.seed(seed)

    colors = np.random.randint(0, 255, size=(num_colors, 3), dtype="uint8")
    colors = np.vstack([colors]).astype("uint8")
    colors = [tuple(color) for color in list(colors)]
    colors = [tuple(int(e) for e in color) for color in colors]

    return colors
