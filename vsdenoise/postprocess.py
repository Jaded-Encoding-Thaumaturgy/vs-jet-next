from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

from vsexprtools import ExprOp, ExprToken, norm_expr
from vskernels import Bilinear
from vsmasktools import EdgeDetectT, FDoGTCanny, range_mask
from vsrgtools import RemoveGrainMode, bilateral, box_blur, gauss_blur, removegrain
from vstools import (
    ColorRange, CustomIndexError, CustomIntEnum, FuncExceptT, InvalidColorFamilyError, KwargsT, PlanesT, check_ref_clip,
    check_variable, fallback, flatten_vnodes, get_y, normalize_planes, scale_value, scale_mask, vs
)

from .fft import DFTTest, fft3d
from .nlm import nl_means

__all__ = [
    'decrease_size',

    'PostProcess'
]


def decrease_size(
    clip: vs.VideoNode, sigmaS: float = 10.0, sigmaR: float = 0.009,
    min_in: int = 180, max_in: int = 230, gamma: float = 1.0,
    mask: vs.VideoNode | tuple[float, float] | tuple[float, float, EdgeDetectT] = (0.0496, 0.125, FDoGTCanny),
    prefilter: bool | tuple[int, int] | float = True, planes: PlanesT = None, show_mask: bool = False, **kwargs: Any
) -> vs.VideoNode:
    """
    Forcibly reduce the required bitrate to encode a clip by blurring away noise and grain
    in areas they won't be visible in.

    Grain and noise in really bright areas can be incredibly hard to spot for even experienced encoders,
    and will eat up a lot of extra bitrate. As this grain is invisible, there's little reason
    to go out of your way to better preserve it, and aq-modes like aq-mode 3 already incentivize
    the encoder to spend more bits in darker areas anyway.

    A gradient mask is used internally to prevent "hard edges" from forming on the boundaries of the mask.
    Additionally, an edgemask is used to prevent clearly-defined detail from being blurred away.

    :param clip:        Clip to process.
    :param sigmaS:      Sigma of Gaussian function to calculate spatial weight.
                        See the `vsrgtools.bilateral` documentation for more information.
                        Default: 10.0.
    :param sigmaR:      Sigma of Gaussian function to calculate range weight.
                        See the `vsrgtools.bilateral` documentation for more information.
                        Default: 0.009.
    :param min_in:      Starting pixel value for the gradient mask. Must be a value between 0–255.
                        Low values are not recommended, as this will start to blur a lot more detail.
                        Default: 180.
    :param max_in:      Ending pixel value for the gradient mask. Must be a value between 0–255.
                        This value must be greater than `min_in`. Any pixel values above this will be fully masked.
                        Default: 230.
    :param mask:        Mask node for masking out details from the blur.
    :param prefilter:   Prefilter the clip prior to masked blurring.

                        If you pass a float, a gauss blur will be used with the value determining its sigma.
                        If you pass a tuple of floats, a box blur will be used.
                        The first value is the radii, and the second is the number of passes.
                        If you pass `True`, it defaults to `box_blur(2, 4)`.
                        Set `False` to disable.

                        Default: True.
    :param planes:      Planes to process. If None, all planes. Default: None.
    :param show_mask:   Return the gradient mask clip. Default: False.
    :param kwargs:      Additional keyword arguments to pass to bilateral.

    :return:            Clip with the brightest areas, as defined by the gradient mask, heavily blurred.

    :raises IndexError:                 `min_in` is greater than `max_in`.
    :raises InvalidColorFamilyError:    Input clip is not a YUV clip.
    :raises InvalidColorFamilyError:    A VideoNode is passed to `mask` and the clip is not a GRAY clip.
    """
    assert check_variable(clip, decrease_size)

    if min_in > max_in:
        raise CustomIndexError('The blur min must be lower than max!', decrease_size, dict(min=min_in, max=max_in))

    InvalidColorFamilyError.check(clip, vs.YUV, decrease_size)

    planes = normalize_planes(clip, planes)

    pre = get_y(clip)

    if isinstance(mask, vs.VideoNode):
        InvalidColorFamilyError.check(mask, vs.GRAY, decrease_size)
        check_ref_clip(pre, mask)  # type: ignore
    else:
        pm_min, pm_max, *emask = mask  # type: ignore

        if pm_min > pm_max:
            raise CustomIndexError('The mask min must be lower than max!', decrease_size, dict(min=pm_min, max=pm_max))

        pm_min = scale_mask(pm_min, 32, clip, range_out=ColorRange.FULL)
        pm_max = scale_mask(pm_max, 32, clip, range_out=ColorRange.FULL)

        yuv444 = Bilinear.resample(
            range_mask(clip, rad=3, radc=2), clip.format.replace(subsampling_h=0, subsampling_w=0)
        )

        mask = FDoGTCanny.ensure_obj(emask[0] if emask else None).edgemask(pre)
        mask = mask.std.Maximum().std.Minimum()

        mask_planes = flatten_vnodes(yuv444, mask, split_planes=True)

        mask = norm_expr(
            mask_planes, f'x y max z max {pm_min} < 0 {ExprToken.RangeMax} ? a max {pm_max} < 0 {ExprToken.RangeMax} ?'
        )

        mask = box_blur(mask, 1, 2)

    if prefilter is True:
        prefilter = (2, 4)

    if prefilter:
        if isinstance(prefilter, tuple):
            pre = box_blur(pre, *prefilter)
        else:
            pre = gauss_blur(pre, prefilter)

    minf = scale_value(min_in, 8, pre)
    maxf = scale_value(max_in, 8, pre)

    mask = norm_expr(
        [pre, mask],  # type: ignore
        f'x {ExprOp.clamp(minf, maxf)} {minf} - {maxf} {minf} - / {1 / gamma} '
        f'pow {ExprOp.clamp(0, 1)} {ExprToken.RangeMax} * y -', planes
    )

    if show_mask:
        return mask

    denoise = bilateral(clip, sigmaS, sigmaR, **kwargs)

    return clip.std.MaskedMerge(denoise, mask, planes)


@dataclass
class PostProcessConfig:
    mode: PostProcess
    kwargs: KwargsT

    _sigma: float | None = None
    _tr: int | None = None
    _block_size: int | None = None
    merge_strength: int = 0

    @property
    def sigma(self) -> float:
        sigma = fallback(self._sigma, 1.0)

        if self.mode is PostProcess.DFTTEST:
            return sigma * 4

        if self.mode is PostProcess.NL_MEANS:
            return sigma / 2

        return sigma

    @property
    def tr(self) -> int:
        if self.mode <= 0:
            return 0

        tr = fallback(self._tr, 1)

        if self.mode is PostProcess.DFTTEST:
            return min(tr, 3)

        if self.mode in {PostProcess.FFT3D_MED, PostProcess.FFT3D_HIGH}:
            return min(tr, 2)

        return tr

    @property
    def block_size(self) -> int:
        if self.mode is PostProcess.DFTTEST:
            from .fft import BackendInfo

            backend_info = BackendInfo.from_param(self.kwargs.pop('plugin', DFTTest.Backend.AUTO))

            if backend_info.resolved_backend.is_dfttest2:
                return 16

        return fallback(self._block_size, [0, 48, 32, 12, 0][self.mode.value])

    def __call__(self, clip: vs.VideoNode, planes: PlanesT = None, func: FuncExceptT | None = None) -> vs.VideoNode:
        func = func or self.__class__

        if self.mode is PostProcess.REPAIR:
            return removegrain(clip, RemoveGrainMode.MINMAX_AROUND1)

        if self.mode in {PostProcess.FFT3D_MED, PostProcess.FFT3D_HIGH}:
            return fft3d(clip, func, bw=self.block_size, bh=self.block_size, bt=self.tr * 2 + 1, **self.kwargs)

        if self.mode is PostProcess.DFTTEST:
            return DFTTest.denoise(
                clip, self.sigma, tr=self.tr, block_size=self.block_size,
                planes=planes, **(KwargsT(overlap=int(self.block_size * 9 / 12)) | self.kwargs)  # type: ignore
            )

        if self.mode is PostProcess.NL_MEANS:
            return nl_means(
                clip, self.sigma, self.tr, planes=planes, **(KwargsT(sr=2) | self.kwargs)  # type: ignore
            )

        return clip


class PostProcess(CustomIntEnum):
    REPAIR = 0
    FFT3D_HIGH = 1
    FFT3D_MED = 2
    DFTTEST = 3
    NL_MEANS = 4

    if TYPE_CHECKING:
        from .postprocess import PostProcess

        @overload
        def __call__(  # type: ignore
            self: Literal[PostProcess.REPAIR], *, merge_strength: int = 0
        ) -> PostProcessConfig:
            ...

        @overload
        def __call__(  # type: ignore
            self: Literal[PostProcess.NL_MEANS], *, sigma: float = 1.0, tr: int | None = None,
            merge_strength: int = 0, **kwargs: Any
        ) -> PostProcessConfig:
            ...

        @overload
        def __call__(
            self, *, sigma: float = 1.0, tr: int | None = None, block_size: int | None = None,
            merge_strength: int = 0, **kwargs: Any
        ) -> PostProcessConfig:
            ...

        def __call__(
            self, *, sigma: float = 1.0, tr: int | None = None, block_size: int | None = None,
            merge_strength: int = 0, **kwargs: Any
        ) -> PostProcessConfig:
            ...
    else:
        def __call__(
            self, *, sigma: float = 1.0, tr: int | None = None, block_size: int | None = None,
            merge_strength: int = 0, **kwargs: Any
        ) -> PostProcessConfig:
            return PostProcessConfig(self, kwargs, sigma, tr, block_size, merge_strength)
