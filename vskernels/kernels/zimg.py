
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jetpytools import inject_kwargs_params
from vstools import Dar, Sar, inject_self, vs

from ..types import BorderHandling, Center, LeftShift, SampleGridModel, Slope, TopShift
from .abstract import Descaler
from .complex import ComplexKernel

__all__ = [
    'ZimgDescaler',
    'ZimgComplexKernel'
]


class ZimgDescaler(Descaler):
    if TYPE_CHECKING:
        @inject_self.cached
        @inject_kwargs_params
        def descale(  # type: ignore[override]
            self, clip: vs.VideoNode, width: int | None, height: int | None, shift: tuple[TopShift, LeftShift] = (0, 0),
            *, blur: float = 1.0, border_handling: BorderHandling = BorderHandling.MIRROR, **kwargs: Any
        ) -> vs.VideoNode:
            ...


class ZimgComplexKernel(ComplexKernel, ZimgDescaler):  # type: ignore[misc]
    if TYPE_CHECKING:
        # Override signature to add `blur` and remove `border_handling`.
        @inject_self.cached
        @inject_kwargs_params
        def scale(  # type: ignore[override]
            self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
            shift: tuple[TopShift, LeftShift] = (0, 0),
            *,
            blur: float = 1.0,
            sample_grid_model: SampleGridModel = SampleGridModel.MATCH_EDGES,
            sar: Sar | bool | float | None = None, dar: Dar | bool | float | None = None, keep_ar: bool | None = None,
            linear: bool = False, sigmoid: bool | tuple[Slope, Center] = False,
            **kwargs: Any
        ) -> vs.VideoNode:
            ...

        @inject_self.cached
        @inject_kwargs_params
        def descale(  # type: ignore[override]
            self, clip: vs.VideoNode, width: int | None, height: int | None, shift: tuple[TopShift, LeftShift] = (0, 0),
            *, blur: float = 1.0, border_handling: BorderHandling, ignore_mask: vs.VideoNode | None = None,
            linear: bool = False, sigmoid: bool | tuple[Slope, Center] = False, **kwargs: Any
        ) -> vs.VideoNode:
            ...

    def get_params_args(
        self, is_descale: bool, clip: vs.VideoNode, width: int | None = None, height: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        args = super().get_params_args(is_descale, clip, width, height, **kwargs)

        if not is_descale:
            for key in ('border_handling', 'ignore_mask', 'force', 'force_h', 'force_v'):
                args.pop(key, None)

        return args
