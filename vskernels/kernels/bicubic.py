from __future__ import annotations

from math import sqrt
from typing import Any

from vstools import CustomValueError, inject_self, core, vs

from .zimg import ZimgComplexKernel

__all__ = [
    'Bicubic',
    'BSpline',
    'Hermite',
    'Mitchell',
    'Catrom',
    'FFmpegBicubic',
    'AdobeBicubic',
    'AdobeBicubicSharper',
    'AdobeBicubicSmoother',
    'BicubicSharp',
    'RobidouxSoft',
    'Robidoux',
    'RobidouxSharp',
    'BicubicAuto',
]


class Bicubic(ZimgComplexKernel):
    """
    Built-in bicubic resizer.

    Default: b=0, c=0.5

    :param b: B-param for bicubic kernel
    :param c: C-param for bicubic kernel
    """

    scale_function = resample_function = core.lazy.resize2.Bicubic
    descale_function = core.lazy.descale.Debicubic

    def __init__(self, b: float = 0, c: float = 1 / 2, **kwargs: Any) -> None:
        self.b = b
        self.c = c
        super().__init__(**kwargs)

    def get_params_args(
        self, is_descale: bool, clip: vs.VideoNode, width: int | None = None, height: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        args = super().get_params_args(is_descale, clip, width, height, **kwargs)
        if is_descale:
            return args | dict(b=self.b, c=self.c)
        return args | dict(filter_param_a=self.b, filter_param_b=self.c)

    @inject_self.cached.property
    def kernel_radius(self) -> int:  # type: ignore[override]
        if (self.b, self.c) == (0, 0):
            return 1
        return 2


class BSpline(Bicubic):
    """Bicubic b=1, c=0"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(b=1, c=0, **kwargs)


class Hermite(Bicubic):
    """Bicubic b=0, c=0"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(b=0, c=0, **kwargs)


class Mitchell(Bicubic):
    """Bicubic b=1/3, c=1/3"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(b=1 / 3, c=1 / 3, **kwargs)


class Catrom(Bicubic):
    """Bicubic b=0, c=0.5"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(b=0, c=1 / 2, **kwargs)


class FFmpegBicubic(Bicubic):
    """Bicubic b=0, c=0.6; FFmpeg's swscale default"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(b=0, c=0.6, **kwargs)


class AdobeBicubic(Bicubic):
    """Bicubic b=0, c=0.75; Adobe's "Bicubic" interpolation preset"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(b=0, c=3 / 4, **kwargs)


class AdobeBicubicSharper(Bicubic):
    """Bicubic b=0, c=1, blur=1.05; Adobe's "Bicubic Sharper" interpolation preset."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(b=0, c=1, blur=1.05, **kwargs)


class AdobeBicubicSmoother(Bicubic):
    """Bicubic b=0, c=0.625, blur=1.15; Adobe's "Bicubic Smoother" interpolation preset."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(b=0, c=5 / 8, blur=1.15, **kwargs)


class BicubicSharp(Bicubic):
    """Bicubic b=0, c=1"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(b=0, c=1, **kwargs)


class RobidouxSoft(Bicubic):
    """Bicubic b=0.67962, c=0.16019"""

    def __init__(self, **kwargs: Any) -> None:
        b = (9 - 3 * sqrt(2)) / 7
        c = (1 - b) / 2
        super().__init__(b=b, c=c, **kwargs)


class Robidoux(Bicubic):
    """Bicubic b=0.37822, c=0.31089"""

    def __init__(self, **kwargs: Any) -> None:
        b = 12 / (19 + 9 * sqrt(2))
        c = 113 / (58 + 216 * sqrt(2))
        super().__init__(b=b, c=c, **kwargs)


class RobidouxSharp(Bicubic):
    """Bicubic b=0.26201, c=0.36899"""

    def __init__(self, **kwargs: Any) -> None:
        b = 6 / (13 + 7 * sqrt(2))
        c = 7 / (2 + 12 * sqrt(2))
        super().__init__(b=b, c=c, **kwargs)


class BicubicAuto(Bicubic):
    """
    Kernel that follows the rule of:
    b + 2c = target
    """

    def __init__(self, b: float | None = None, c: float | None = None, **kwargs: Any) -> None:
        if None not in {b, c}:
            raise CustomValueError("You can't specify both b and c!", self.__class__)

        self.b, self.c = self._get_bc_args(b, c)

        super().__init__(**kwargs)

    def _get_bc_args(self, b: float | None, c: float | None) -> tuple[float, float]:
        autob = 0.0 if b is None else b
        autoc = 0.5 if c is None else c

        if c is not None and b is None:
            autob = 1.0 - 2 * c
        elif c is None and b is not None:
            autoc = (1.0 - b) / 2

        return autob, autoc
