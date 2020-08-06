#!/usr/bin/env python
import math
import unittest
from rail_marking.utils.math_utils import *


class MainTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(MainTest, self).__init__(*args, **kwargs)

    def test_binomial_coeff(self):
        self.assertEqual(int(estimate_binomial_coeff(10, 5)), 252)
        self.assertEqual(int(estimate_binomial_coeff(30, 17)), 119759850)


if __name__ == "__main__":
    unittest.main()
