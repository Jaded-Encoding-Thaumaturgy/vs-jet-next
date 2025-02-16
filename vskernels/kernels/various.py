from __future__ import annotations

from math import cos, exp, log, pi, sqrt
from typing import Any

from vstools import inject_self

from .complex import CustomComplexKernel, CustomComplexTapsKernel
from .helpers import sinc

__all__ = [
    'Gaussian',
    'Box',
    'BlackMan',
    'BlackManMinLobe',
    'Sinc',
    'Hann',
    'Hamming',
    'Welch',
    'Bohman',
    'Cosine',
]


class gauss_sigma(float):
    def from_fmtc(self, curve: float) -> float:
        if not curve:
            return 0.0
        return sqrt(1.0 / (2.0 * (curve / 10.0) * log(2)))

    def to_fmtc(self, sigma: float) -> float:
        if not sigma:
            return 0.0
        return 10 / (2 * log(2) * (sigma ** 2))

    def from_libplacebo(self, sigma: float) -> float:
        if not sigma:
            return 0.0
        return sqrt(sigma / 4)

    def to_libplacebo(self, sigma: float) -> float:
        if not sigma:
            return 0.0
        return 4 * (sigma ** 2)


class Gaussian(CustomComplexTapsKernel):
    """Gaussian resizer."""

    def __init__(self, sigma: float = 0.5, taps: float = 2, **kwargs: Any) -> None:
        """Sigma is the same as imagemagick's sigma scaling."""

        self._sigma = sigma

        super().__init__(taps, **kwargs)

    @inject_self.property
    def sigma(self) -> gauss_sigma:
        return gauss_sigma(self._sigma)

    @inject_self.cached
    def kernel(self, *, x: float) -> float:
        return 1 / (self._sigma * sqrt(2 * pi)) * exp(-x ** 2 / (2 * self._sigma ** 2))


class Box(CustomComplexKernel):
    """Box resizer."""

    _static_kernel_radius = 1

    @inject_self.cached
    def kernel(self, *, x: float) -> float:
        return 1.0 if x >= -0.5 and x < 0.5 else 0.0


class BlackMan(CustomComplexTapsKernel):
    """Blackman resizer."""

    def __init__(self, taps: float = 4, **kwargs: Any) -> None:
        super().__init__(taps, **kwargs)

    def _win_coef(self, x: float) -> float:
        w_x = x * (pi / self.kernel_radius)

        return 0.42 + 0.50 * cos(w_x) + 0.08 * cos(w_x * 2)

    @inject_self.cached
    def kernel(self, *, x: float) -> float:
        if x >= self.kernel_radius:
            return 0.0

        return sinc(x) * self._win_coef(x)


class BlackManMinLobe(BlackMan):
    """Blackmanminlobe resizer."""

    def _win_coef(self, x: float) -> float:
        w_x = x * (pi / self.kernel_radius)

        return 0.355768 + 0.487396 * cos(w_x) + 0.144232 * cos(w_x * 2) + 0.012604 * cos(w_x * 3)


class Sinc(CustomComplexTapsKernel):
    """Sinc resizer."""

    def __init__(self, taps: float = 4, **kwargs: Any) -> None:
        super().__init__(taps, **kwargs)

    @inject_self.cached
    def kernel(self, *, x: float) -> float:
        if x >= self.kernel_radius:
            return 0.0

        return sinc(x)


class Hann(CustomComplexTapsKernel):
    """Hann kernel."""

    @inject_self.cached
    def kernel(self, *, x: float) -> float:
        if x >= self.kernel_radius:
            return 0.0

        return 0.5 + 0.5 * cos(pi * x)


class Hamming(CustomComplexTapsKernel):
    """Hamming kernel."""

    @inject_self.cached
    def kernel(self, *, x: float) -> float:
        if x >= self.kernel_radius:
            return 0.0

        return 0.54 + 0.46 * cos(pi * x)


class Welch(CustomComplexTapsKernel):
    """Welch kernel."""

    @inject_self.cached
    def kernel(self, *, x: float) -> float:
        if abs(x) >= 1.0:
            return 0.0

        return 1.0 - x * x


class Cosine(CustomComplexTapsKernel):
    """Cosine kernel."""

    @inject_self.cached
    def kernel(self, *, x: float) -> float:
        if x >= self.kernel_radius:
            return 0.0

        cosine = cos(pi * x)

        return 0.34 + cosine * (0.5 + cosine * 0.16)


class Bohman(CustomComplexTapsKernel):
    """Bohman kernel."""

    @inject_self.cached
    def kernel(self, *, x: float) -> float:
        if x >= self.kernel_radius:
            return 0.0

        cosine = cos(pi * x)
        sine = sqrt(1.0 - cosine * cosine)

        return (1.0 - x) * cosine + (1.0 / pi) * sine
