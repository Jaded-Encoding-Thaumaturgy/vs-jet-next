from __future__ import annotations

from math import pi, sin

__all__ = [
    'sinc', 'poly3'
]


def sinc(x: float) -> float:
    return 1.0 if x == 0.0 else sin(x * pi) / (x * pi)


def poly3(x: float, c0: float, c1: float, c2: float, c3: float) -> float:
    return c0 + x * (c1 + x * (c2 + x * c3))
