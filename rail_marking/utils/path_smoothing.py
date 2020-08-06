#!/usr/bin/env python
import math
from .math_utils import estimate_binomial_coeffs, estimate_polynomial_coeffs


__all__ = ["BezierCurve2D"]


class BezierCurve2D(object):
    def __init__(self, control_points: list, downscale_length_ratio=20, eps=1e-9):
        self._control_points = control_points
        self._downscale_length_ratio = downscale_length_ratio
        self._eps = 1e-9

        if self.degree() < 1:
            raise Exception("degree must be more than 1\n")

        self._binomial_coeffs = estimate_binomial_coeffs(self.degree())

    def estimate_trajectory(self, num_points=None, start=0.0, end=1.0) -> list:
        if not num_points:
            num_points = self._control_points_length() // self._downscale_length_ratio
        else:
            num_points = math.max(
                self._control_points_length() // self._downscale_length_ratio,
                num_points,
            )

        trajectory = [None] * (num_points + 1)
        delta = (end - start) / num_points
        for i in range(num_points + 1):
            t = start + i * delta
            trajectory[i] = self._point_at(t)

        return trajectory

    def degree(self) -> int:
        return len(self._control_points) - 1

    def _value_at(self, axis: int, t: float) -> float:
        # x, y dimension
        if axis < 0 or axis > 2:
            raise Exception("invalid axis")

        _polynomial_coeffs = estimate_polynomial_coeffs(self.degree(), t)
        sum = 0.0
        for i in range(self.degree() + 1):
            sum += self._control_points[i][axis] * self._binomial_coeffs[i] * _polynomial_coeffs[i]

        return sum

    def _point_at(self, t: float) -> list:
        output_point = [0.0, 0.0]
        output_point[0] = self._value_at(0, t)
        output_point[1] = self._value_at(1, t)
        return output_point

    def _control_points_length(self) -> int:
        length = 0.0
        for i in range(self.degree()):
            length += math.sqrt(
                math.pow(self._control_points[i][0] - self._control_points[i + 1][0], 2)
                + math.pow(self._control_points[i][1] - self._control_points[i + 1][1], 2)
            )

        return int(length)
