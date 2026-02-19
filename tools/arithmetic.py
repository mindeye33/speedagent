"""Small numeric utilities.

Kept intentionally general; these are mostly for testing/tooling.
"""

from __future__ import annotations

from typing import Iterable


def add(x: float, y: float) -> float:
    """Add two numbers."""

    return float(x + y)


def multiply(x: float, y: float) -> float:
    """Multiply two numbers."""

    return float(x * y)


def power(x: float, y: float) -> float:
    """Raise x to the power y (x**y)."""

    return float(x**y)


def dot(a: Iterable[float], b: Iterable[float]) -> float:
    """Dot product of two 1D iterables."""

    s = 0.0
    for x, y in zip(a, b):
        s += float(x) * float(y)
    return float(s)


def mean(xs: Iterable[float]) -> float:
    """Mean of an iterable of numbers."""

    xs = list(xs)
    if not xs:
        raise ValueError("mean() requires at least one value")
    return float(sum(float(x) for x in xs) / len(xs))
