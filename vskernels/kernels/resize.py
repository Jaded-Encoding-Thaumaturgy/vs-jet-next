from __future__ import annotations

from math import ceil
from typing import Any

from vstools import core, inject_self, vs

from .zimg import ZimgComplexKernel

__all__ = [
    'Point',
    'Bilinear',
    'Lanczos',
]


class Point(ZimgComplexKernel):
    """Built-in point resizer."""

    scale_function = resample_function = descale_function = core.lazy.resize2.Point
    _static_kernel_radius = 1


class Bilinear(ZimgComplexKernel):
    """Built-in bilinear resizer."""

    scale_function = resample_function = core.lazy.resize2.Bilinear
    descale_function = core.lazy.descale.Debilinear
    _static_kernel_radius = 1


class Lanczos(ZimgComplexKernel):
    """
    Built-in lanczos resizer.

    Dependencies:

    * VapourSynth-descale

    :param taps: taps param for lanczos kernel
    """

    scale_function = resample_function = core.lazy.resize2.Lanczos
    descale_function = core.lazy.descale.Delanczos

    def __init__(self, taps: int = 3, **kwargs: Any) -> None:
        self.taps = taps
        super().__init__(**kwargs)

    def get_params_args(
        self, is_descale: bool, clip: vs.VideoNode, width: int | None = None, height: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        args = super().get_params_args(is_descale, clip, width, height, **kwargs)
        if is_descale:
            return args | dict(taps=self.taps)
        return args | dict(filter_param_a=self.taps)

    @inject_self.property
    def kernel_radius(self) -> int: # type: ignore[override]
        return ceil(self.taps)
