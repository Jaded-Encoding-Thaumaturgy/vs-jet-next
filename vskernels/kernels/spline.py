from __future__ import annotations

from math import comb
from typing import Any

from vstools import core, inject_self

from .complex import CustomComplexTapsKernel
from .helpers import poly3
from .zimg import ZimgComplexKernel

__all__ = [
    'Spline',
    'Spline16',
    'Spline36',
    'Spline64',
]


class Spline(CustomComplexTapsKernel):
    """Spline resizer."""

    def __init__(self, taps: float = 2, **kwargs: Any) -> None:
        super().__init__(taps, **kwargs)

        if hasattr(self, '_static_coeffs'):
            self._coefs = self._static_coeffs
        else:
            self._coefs = self._splineKernelCoeff()

    def _naturalCubicSpline(self, values: list[int]) -> list[float]:
        import numpy as np

        n = len(values) - 1

        rhs = values[:-1] + values[1:] + [0] * (2 * n)

        eqns = []
        # left value = sample
        eqns += [[0] * (4 * i) + [i ** 3, i ** 2, i, 1] + [0] * (4 * (n - i - 1)) for i in range(n)]
        # right value = sample
        eqns += [[0] * (4 * i) + [(i + 1) ** 3, (i + 1) ** 2, i + 1, 1] + [0] * (4 * (n - i - 1)) for i in range(n)]
        # derivatives match
        eqns += [
            (
                [0] * (4 * i) +  # noqa: W504
                [3 * (i + 1) ** 2, 2 * (i + 1), 1, 0] +  # noqa: W504
                [-3 * (i + 1) ** 2, -2 * (i + 1), -1, 0] +  # noqa: W504
                [0] * (4 * (n - i - 2))
            )
            for i in range(n - 1)
        ]
        # second derivatives match
        eqns += [
            [0] * (4 * i) + [6 * (i + 1), 2, 0, 0] + [-6 * (i + 1), -2, 0, 0] + [0] * (4 * (n - i - 2))
            for i in range(n - 1)
        ]
        eqns += [[0, 2, 0, 0] + [0] * (4 * (n - 1))]
        eqns += [[0] * (4 * (n - 1)) + [6 * n ** 2, 2 * n, 0, 0]]

        assert (len(rhs) == len(eqns))

        return list(np.linalg.solve(np.array(eqns), np.array(rhs)))

    def _splineKernelCoeff(self) -> list[float]:
        taps = self.kernel_radius

        coeffs = list[float]()

        def _shiftPolynomial(coeffs: list[float], shift: float) -> list[float]:
            return [
                sum(c * comb(k, m) * (-shift) ** max(0, k - m) for k, c in enumerate(coeffs[::-1]))
                for m in range(len(coeffs))
            ][::-1]

        for i in range(taps):
            samplept = taps - i - 1
            samples = [0] * samplept + [1] + [0] * (2 * taps - samplept - 1)

            assert len(samples) == 2 * taps

            coeffs += _shiftPolynomial(
                self._naturalCubicSpline(samples)[4 * taps - 4:4 * taps], -(taps - 1) + i
            )

        return coeffs

    @inject_self.cached
    def kernel(self, *, x: float) -> float:
        x, taps = abs(x), self.kernel_radius

        if x >= taps:
            return 0.0

        tap = int(x)

        a, b, c, d = self._coefs[4 * tap:4 * tap + 4]

        return poly3(x, d, c, b, a)


class Spline16(ZimgComplexKernel):
    """Spline16 resizer."""

    scale_function = resample_function = core.lazy.resize2.Spline16
    descale_function = core.lazy.descale.Despline16
    _static_kernel_radius = 2


class Spline36(ZimgComplexKernel):
    """Spline36 resizer."""

    scale_function = resample_function = core.lazy.resize2.Spline36
    descale_function = core.lazy.descale.Despline36
    _static_kernel_radius = 3


class Spline64(ZimgComplexKernel):
    """Spline64 resizer."""

    scale_function = resample_function = core.lazy.resize2.Spline64
    descale_function = core.lazy.descale.Despline64
    _static_kernel_radius = 4
