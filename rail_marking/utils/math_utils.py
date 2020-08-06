#!/usr/bin/env python


__all__ = [
    "estimate_binomial_coeff",
    "estimate_binomial_coeffs",
    "estimate_polynomial_coeffs",
]


def estimate_binomial_coeff(n: int, k: int) -> float:
    if n < k:
        raise Exception("n must be less than k\n")

    result = 1.0
    optimized_k = k
    if k > n - k:
        optimized_k = n - k

    for i in range(optimized_k):
        result = (n - i) * result / (i + 1)

    return result


def estimate_binomial_coeffs(n: int) -> list:
    results = [0.0] * (n + 1)

    center = int(n / 2)
    for i in range(center + 1):
        results[i] = estimate_binomial_coeff(n, i)
        results[n - i] = results[i]

    return results


def estimate_polynomial_coeffs(n: int, t: float) -> list:
    import math

    results = [0.0] * (n + 1)
    for i in range(n + 1):
        results[i] = math.pow(1 - t, n - i) * math.pow(t, i)

    return results
