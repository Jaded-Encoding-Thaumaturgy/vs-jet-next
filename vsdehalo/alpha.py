from __future__ import annotations

from typing import Any, Sequence, final

from vsaa import Nnedi3
from vsdenoise import Prefilter
from vsexprtools import ExprOp, combine, norm_expr
from vskernels import Bilinear, BSpline, Lanczos, Mitchell, NoShift, Point, Scaler, ScalerT
from vsmasktools import EdgeDetect, Morpho, RadiusT, Robinson3, XxpandMode, grow_mask, retinex
from vsrgtools import (
    BlurMatrixBase, RemoveGrainMode, RepairMode, box_blur, contrasharpening,
    contrasharpening_dehalo, gauss_blur, limit_filter, repair
)
from vstools import (
    ConvMode, CustomIndexError, CustomIntEnum, CustomValueError, FieldBased, FuncExceptT,
    FunctionUtil, InvalidColorFamilyError, KwargsT, OneDimConvModeT, PlanesT,
    UnsupportedFieldBasedError, check_ref_clip, check_variable, check_variable_format, clamp,
    cround, fallback, get_peak_value, get_y, join, limiter, mod4, normalize_planes, normalize_seq,
    scale_mask, split, to_arr, vs
)

__all__ = [
    'fine_dehalo',
    'fine_dehalo2',
    'dehalo_alpha',
    'dehalo_sigma',
    'dehalomicron',
    'dehalo_merge'
]


FloatIterArr = float | list[float] | tuple[float | list[float], ...]


def _limit_dehalo(
    clip: vs.VideoNode, ref: vs.VideoNode, darkstr: float | list[float], brightstr: float | list[float], planes: PlanesT
) -> vs.VideoNode:
    return norm_expr(
        [clip, ref], 'x y < x x y - {darkstr} * - x x y - {brightstr} * - ?', planes,
        darkstr=darkstr, brightstr=brightstr
    )


def _dehalo_mask(
    clip: vs.VideoNode, ref: vs.VideoNode, lowsens: list[float], highsens: list[float],
    sigma_mask: float | bool, mask_radius: RadiusT, mask_coords: Sequence[int] | None,
    planes: PlanesT
) -> vs.VideoNode:
    peak = get_peak_value(clip)

    mask = norm_expr(
        [
            Morpho.gradient(clip, mask_radius, planes=planes, coords=mask_coords),
            Morpho.gradient(ref, mask_radius, planes=planes, coords=mask_coords)
        ],
        'x 0 = 0.0 x y - x / ? {lowsens} - x {peak} / 256 255 / + 512 255 / / {highsens} + * '
        '0.0 max 1.0 min {peak} *', planes, peak=peak,
        lowsens=[lo / 255 for lo in lowsens], highsens=[hi / 100 for hi in highsens]
    )

    if sigma_mask is not False:
        if sigma_mask is True:
            sigma_mask = 0.0

        conv_values = [float((sig_mask := bool(sigma_mask)))] * 9
        conv_values[4] = 1 / clamp(sigma_mask, 0, 1) if sig_mask else 1

        mask = mask.std.Convolution(conv_values, planes=planes)

    return mask


def _dehalo_schizo_norm(*values: FloatIterArr) -> list[tuple[list[float], ...]]:
    iterations = max([(len(x) if isinstance(x, tuple) else 1) for x in values])

    return zip(*[  # type: ignore
        tuple(normalize_seq(x) for x in y)
        for y in [
            (*x, *((x[-1], ) * (len(x) - iterations))) if isinstance(x, tuple) else ((x, ) * iterations)
            for x in values
        ]
    ])


def _dehalo_supersample_minmax(
    clip: vs.VideoNode, ref: vs.VideoNode, ss: list[float], supersampler: ScalerT, supersampler_ref: ScalerT,
    planes: PlanesT, func: FuncExceptT
) -> vs.VideoNode:
    supersampler = Scaler.ensure_obj(supersampler, func)
    supersampler_ref = Scaler.ensure_obj(supersampler_ref, func)

    def _supersample(work_clip: vs.VideoNode, dehalo: vs.VideoNode, ss: float) -> vs.VideoNode:
        if ss <= 1.0:
            return repair(work_clip, dehalo, 1)

        w, h = mod4(work_clip.width * ss), mod4(work_clip.height * ss)
        ss_clip = norm_expr([
            supersampler.scale(work_clip, w, h),
            supersampler_ref.scale(dehalo.std.Maximum(), w, h),
            supersampler_ref.scale(dehalo.std.Minimum(), w, h)
        ], 'x y min z max', planes)

        return supersampler.scale(ss_clip, work_clip.width, work_clip.height)

    if len(set(ss)) == 1 or planes == [0] or clip.format.num_planes == 1:  # type: ignore
        dehalo = _supersample(clip, ref, ss[0])
    else:
        dehalo = join([
            _supersample(wplane, dplane, ssp) for wplane, dplane, ssp in zip(split(clip), split(ref), ss)
        ])

    return dehalo


@final
class FineDehaloMask(CustomIntEnum):
    MAIN = 1
    EDGES = 3
    SHARP_EDGES = 4
    LARGE_EDGES = 6
    IGNORE_DETAILS = 5
    SHRINK = 2
    SHRINK_EDGES_EXCL = 7


class _fine_dehalo:
    @property
    def Masks(self) -> type[FineDehaloMask]:
        return FineDehaloMask

    def __call__(
        self, clip: vs.VideoNode, rx: FloatIterArr = 2.0, ry: FloatIterArr | None = None, darkstr: FloatIterArr = 0.0,
        brightstr: FloatIterArr = 1.0, lowsens: FloatIterArr = 50.0, highsens: FloatIterArr = 50.0,
        thmi: int = 80, thma: int = 128, thlimi: int = 50, thlima: int = 100, sigma_mask: float | bool = False,
        ss: FloatIterArr = 1.5, contra: int | float | bool = 0.0, exclude: bool = True,
        edgeproc: float = 0.0, edgemask: EdgeDetect = Robinson3(), planes: PlanesT = 0,
        show_mask: int | FineDehaloMask | bool = False,
        mask_radius: RadiusT = 1, downscaler: ScalerT = Mitchell, upscaler: ScalerT = BSpline,
        supersampler: ScalerT = Lanczos(3), supersampler_ref: ScalerT = Mitchell, pre_ss: float = 1.0,
        pre_supersampler: ScalerT = Nnedi3(0, field=0, shifter=NoShift), pre_downscaler: ScalerT = Point,
        mask_coords: Sequence[int] | None = None,
        func: FuncExceptT | None = None
    ) -> vs.VideoNode:
        """
        Halo removal script that uses ``dehalo_alpha`` with a few masks and optional contra-sharpening
        to try removing halos without nuking important details like line edges.

        **For ``rx``, ``ry``, only the first value will be used for calculating the mask.**

        ``rx``, ``ry``, ``darkstr``, ``brightstr``, ``lowsens``, ``highsens``, ``ss`` are all
        configurable per plane and iteration. `tuple` means iteration, `list` plane.

        `rx=(2.0, [2.0, 2.4], [2.2, 2.0, 2.1])` means three iterations.
        * 1st => 2.0 for all planes
        * 2nd => 2.0 for luma, 2.4 for chroma
        * 3rd => 2.2 for luma, 2.0 for u, 2.1 for v


        :param clip:                Source clip.
        :param rx:                  Horizontal radius for halo removal.
        :param ry:                  Vertical radius for halo removal.
        :param darkstr:             Strength factor for dark halos.
        :param brightstr:           Strength factor for bright halos.
        :param lowsens:             Sensitivity setting for how weak the dehalo has to be to get fully accepted.
        :param highsens:            Sensitivity setting for how strong the dehalo has to be to get fully discarded.
        :param thmi:                Minimum threshold for sharp edges; keep only the sharpest edges (line edges).
        :param thma:                Maximum threshold for sharp edges; keep only the sharpest edges (line edges).
        :param thlimi:              Minimum limiting threshold; include more edges than previously ignored details.
        :param thlima:              Maximum limiting threshold; include more edges than previously ignored details.
        :param sigma_mask:          Blurring strength for the mask.
        :param ss:                  Supersampling factor, to avoid creation of aliasing.
        :param contra:              Contrasharpening. If True or int, will use :py:func:`contrasharpening`
                                    otherwise uses :py:func:`contrasharpening_fine_dehalo` with specified level.
        :param exclude:             If True, add an addionnal step to exclude edges close to each other
        :param edgeproc:            If > 0, it will add the edgemask to the processing, defaults to 0.0
        :param edgemask:            Internal mask used for detecting the edges, defaults to Robinson3()
        :param planes:              Planes to process.
        :param show_mask:           Whether to show the computed halo mask. 1-7 values to select intermediate masks.
        :param mask_radius:         Mask expanding radius with ``gradient``.
        :param downscaler:          Scaler used to downscale the clip.
        :param upscaler:            Scaler used to upscale the downscaled clip.
        :param supersampler:        Scaler used to supersampler the rescaled clip to `ss` factor.
        :param supersampler_ref:    Reference scaler used to clamp the supersampled clip. Has to be blurrier.
        :param pre_ss:              Supersampling rate used before anything else.
        :param pre_supersampler:    Supersampler used for ``pre_ss``.
        :param pre_downscaler:      Downscaler used for undoing the upscaling done by ``pre_supersampler``.
        :param func:                Function from where this function was called.

        :return:                    Dehaloed clip.


        """
        func = func or 'fine_dehalo'

        assert check_variable(clip, func)

        if FieldBased.from_video(clip).is_inter:
            raise UnsupportedFieldBasedError('Only progressive video is supported!', func)

        InvalidColorFamilyError.check(clip, (vs.GRAY, vs.YUV), func)

        if show_mask is not False and not (0 < int(show_mask) <= 7):
            raise CustomValueError('valid values for show_mask are 1–7!', func)

        thmif, thmaf, thlimif, thlimaf = [
            scale_mask(x, 8, clip)
            for x in [thmi, thma, thlimi, thlima]
        ]

        peak = get_peak_value(clip)
        planes = normalize_planes(clip, planes)

        rx_i, ry_i = cround(to_arr(to_arr(rx)[0])[0]), cround(to_arr(to_arr(fallback(ry, rx))[0])[0])  # type: ignore

        work_clip, *chroma = split(clip) if planes == [0] else (clip, )

        # Main edges #
        # Basic edge detection, thresholding will be applied later.
        edges = edgemask.edgemask(work_clip)

        # Keeps only the sharpest edges (line edges)
        strong = norm_expr(edges, f'x {thmif} - {thmaf - thmif} / {peak} *', planes)

        # Extends them to include the potential halos
        large = Morpho.expand(strong, rx_i, ry_i, planes=planes)

        # Exclusion zones #
        # When two edges are close from each other (both edges of a single
        # line or multiple parallel color bands), the halo removal
        # oversmoothes them or makes seriously bleed the bands, producing
        # annoying artifacts. Therefore we have to produce a mask to exclude
        # these zones from the halo removal.

        # Includes more edges than previously, but ignores simple details
        light = norm_expr(edges, f'x {thlimif} - {thlimaf - thlimif} / {peak} *', planes)

        # To build the exclusion zone, we make grow the edge mask, then shrink
        # it to its original shape. During the growing stage, close adjacent
        # edge masks will join and merge, forming a solid area, which will
        # remain solid even after the shrinking stage.
        # Mask growing
        shrink = Morpho.expand(light, rx_i, ry_i, XxpandMode.ELLIPSE, planes=planes)

        # At this point, because the mask was made of a shades of grey, we may
        # end up with large areas of dark grey after shrinking. To avoid this,
        # we amplify and saturate the mask here (actually we could even
        # binarize it).
        shrink = norm_expr(shrink, 'x 4 *', planes)
        shrink = Morpho.inpand(shrink, rx_i, rx_i, XxpandMode.ELLIPSE, planes=planes)

        # This mask is almost binary, which will produce distinct
        # discontinuities once applied. Then we have to smooth it.
        shrink = box_blur(shrink, 1, 2, planes=planes)

        # Final mask building #

        # Previous mask may be a bit weak on the pure edge side, so we ensure
        # that the main edges are really excluded. We do not want them to be
        # smoothed by the halo removal.
        shr_med = combine([strong, shrink], ExprOp.MAX, planes=planes) if exclude else strong

        # Subtracts masks and amplifies the difference to be sure we get 255
        # on the areas to be processed.
        mask = norm_expr([large, shr_med], 'x y - 2 *', planes)

        # If edge processing is required, adds the edgemask
        if edgeproc > 0:
            mask = norm_expr([mask, strong], f'x y {edgeproc} 0.66 * * +', planes)

        # Smooth again and amplify to grow the mask a bit, otherwise the halo
        # parts sticking to the edges could be missed.
        # Also clamp to legal ranges
        mask = box_blur(mask, planes=planes)

        mask = norm_expr(mask, f'x 2 * {ExprOp.clamp(0, peak)}', planes)

        # Masking #
        if show_mask:
            return [mask, shrink, edges, strong, light, large, shr_med][int(show_mask) - 1]

        dehaloed = dehalo_alpha(
            work_clip, rx, ry, darkstr, brightstr, lowsens, highsens, sigma_mask, ss, planes, False, mask_radius,
            downscaler, upscaler, supersampler, supersampler_ref, pre_ss, pre_supersampler, pre_downscaler,
            mask_coords, func
        )

        if contra:
            if isinstance(contra, float):
                dehaloed = contrasharpening_dehalo(dehaloed, work_clip, contra, planes=planes)
            else:
                dehaloed = contrasharpening(
                    dehaloed, work_clip, int(contra), planes=planes
                )

        y_merge = work_clip.std.MaskedMerge(dehaloed, mask, planes)

        if chroma:
            return join([y_merge, *chroma], clip.format.color_family)

        return y_merge

    def mask(
        self, clip: vs.VideoNode, dehaloed: vs.VideoNode | None = None,
        rx: int = 1, ry: int | None = None, thmi: int = 50, thma: int = 100, thlimi: int = 50, thlima: int = 100,
        exclude: bool = True, edgeproc: float = 0.0, edgemask: EdgeDetect = Robinson3(), pre_ss: int = 1,
        pre_supersampler: ScalerT = Bilinear,
        mask: int | FineDehaloMask = 1, planes: PlanesT = 0, first_plane: bool = False, func: FuncExceptT | None = None
    ) -> vs.VideoNode:
        """
        The fine_dehalo mask.

        :param clip:                Source clip.
        :param dehaloed:            Optional dehaloed clip to mask. If this is specified, instead of returning
                                    the mask, the function will return the maskedmerged clip to this.
        :param rx:                  Horizontal radius for halo removal.
        :param ry:                  Vertical radius for halo removal.
        :param thmi:                Minimum threshold for sharp edges; keep only the sharpest edges (line edges).
        :param thma:                Maximum threshold for sharp edges; keep only the sharpest edges (line edges).
        :param thlimi:              Minimum limiting threshold; include more edges than previously ignored details.
        :param thlima:              Maximum limiting threshold; include more edges than previously ignored details.
        :param exclude:             If True, add an addionnal step to exclude edges close to each other
        :param edgeproc:            If > 0, it will add the edgemask to the processing, defaults to 0.0
        :param edgemask:            Internal mask used for detecting the edges, defaults to Robinson3()
        :param get_mask:            Whether to show the computed halo mask. 1-7 values to select intermediate masks.
        :param pre_ss:              Supersampling rate used before anything else. This value will be be rounded.
        :param pre_supersampler:    Supersampler used for ``pre_ss``.
        :param planes:              Planes to process.
        :param first_plane:         Whether to mask chroma planes with luma mask.
        :param func:                Function from where this function was called.

        :return:                    Mask or masked clip.
        """
        work_clip = get_y(clip)

        if pre_ss > 1.0:
            pre_ss = max(round(pre_ss), 2)

            work_clip = Scaler.ensure_obj(pre_supersampler, func).scale(
                work_clip, work_clip.width * pre_ss, work_clip.height * pre_ss,
                (-(0.5 / pre_ss), -(0.5 / pre_ss))
            )

        dehalo_mask = self(
            work_clip, rx, ry, thmi=thmi, thma=thma, thlimi=thlimi, thlima=thlima, exclude=exclude,
            edgeproc=edgeproc, edgemask=edgemask, planes=planes, show_mask=mask, func=func or self.mask
        )

        if (dehalo_mask.width, dehalo_mask.height) != (clip.width, clip.height):
            dehalo_mask = Point.scale(dehalo_mask, clip.width, clip.height)

        if dehaloed:
            return clip.std.MaskedMerge(dehaloed, dehalo_mask, planes, first_plane)

        return dehalo_mask


fine_dehalo = _fine_dehalo()


def fine_dehalo2(
    clip: vs.VideoNode,
    mode: OneDimConvModeT = ConvMode.HV,
    radius: int = 2, mask_radius: int = 2,
    brightstr: float = 1.0, darkstr: float = 1.0,
    dark: bool | None = True,
    show_mask: bool = False
) -> vs.VideoNode:
    """
    Halo removal function for 2nd order halos.

    :param clip:            Source clip.
    :param mode:            Horizontal/Vertical or both ways.
    :param radius:          Radius for the fixing convolution.
    :param mask_radius:     Radius for mask growing.
    :param brightstr:       Strength factor for bright halos.
    :param darkstr:         Strength factor for dark halos.
    :param dark:            Whether to filter for dark or bright haloing.
                            None for disable merging with source clip.
    :param show_mask:       Whether to return the computed mask.

    :return:            Dehaloed clip.
    """
    func = fine_dehalo2

    assert clip.format

    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise ValueError('fine_dehalo2: format not supported')

    work_clip, *chroma = split(clip)

    mask_h = mask_v = None

    if mode in {ConvMode.HV, ConvMode.VERTICAL}:
        mask_h = BlurMatrixBase([1, 2, 1, 0, 0, 0, -1, -2, -1], ConvMode.V)(work_clip, divisor=4, saturate=False)

    if mode in {ConvMode.HV, ConvMode.HORIZONTAL}:
        mask_v = BlurMatrixBase([1, 0, -1, 2, 0, -2, 1, 0, -1], ConvMode.H)(work_clip, divisor=4, saturate=False)

    if mask_h and mask_v:
        mask_h2 = norm_expr([mask_h, mask_v], ['x 3 * y -', ExprOp.clamp()], func=func)
        mask_v2 = norm_expr([mask_v, mask_h], ['x 3 * y -', ExprOp.clamp()], func=func)
        mask_h, mask_v = mask_h2, mask_v2
    elif mask_h:
        mask_h = norm_expr(mask_h, ['x 3 *', ExprOp.clamp()], func=func)
    elif mask_v:
        mask_v = norm_expr(mask_v, ['x 3 *', ExprOp.clamp()], func=func)

    if mask_h:
        mask_h = grow_mask(mask_h, mask_radius, coord=[0, 1, 0, 0, 0, 0, 1, 0], multiply=1.8, func=func)
    if mask_v:
        mask_v = grow_mask(mask_v, mask_radius, coord=[0, 0, 0, 1, 1, 0, 0, 0], multiply=1.8, func=func)

    if clip.format.sample_type == vs.FLOAT:
        mask_h = mask_h and limiter(mask_h, func=func)
        mask_v = mask_v and limiter(mask_v, func=func)

    if show_mask:
        if mask_h and mask_v:
            return combine([mask_h, mask_v], ExprOp.MAX)

        assert (ret_mask := mask_h or mask_v)

        return ret_mask

    fix_weights = list(range(-1, -radius - 1, -1))
    fix_rweights = list(reversed(fix_weights))
    fix_zeros, fix_mweight = [0] * radius, 10 * (radius + 2)

    fix_h_conv = [*fix_weights, *fix_zeros, fix_mweight, *fix_zeros, *fix_rweights]
    fix_v_conv = [*fix_rweights, *fix_zeros, fix_mweight, *fix_zeros, *fix_weights]

    fix_h = ExprOp.convolution('x', fix_h_conv, mode=ConvMode.HORIZONTAL)(work_clip, func=func)
    fix_v = ExprOp.convolution('x', fix_v_conv, mode=ConvMode.VERTICAL)(work_clip, func=func)

    dehaloed = work_clip

    for fix, mask in [(fix_h, mask_v), (fix_v, mask_h)]:
        if mask:
            dehaloed = dehaloed.std.MaskedMerge(fix, mask)

    if dark is not None:
        dehaloed = combine([work_clip, dehaloed], ExprOp.MAX if dark else ExprOp.MIN)

    if darkstr != brightstr != 1.0:
        dehaloed = _limit_dehalo(work_clip, dehaloed, darkstr, brightstr, 0)

    if not chroma:
        return dehaloed

    return join([dehaloed, *chroma], clip.format.color_family)


def dehalo_alpha(
    clip: vs.VideoNode, rx: FloatIterArr = 2.0, ry: FloatIterArr | None = None, darkstr: FloatIterArr = 0.0,
    brightstr: FloatIterArr = 1.0, lowsens: FloatIterArr = 50.0, highsens: FloatIterArr = 50.0,
    sigma_mask: float | bool = False, ss: FloatIterArr = 1.5, planes: PlanesT = 0, show_mask: bool = False,
    mask_radius: RadiusT = 1, downscaler: ScalerT = Mitchell, upscaler: ScalerT = BSpline,
    supersampler: ScalerT = Lanczos(3), supersampler_ref: ScalerT = Mitchell, pre_ss: float = 1.0,
    pre_supersampler: ScalerT = Nnedi3(0, field=0, shifter=NoShift), pre_downscaler: ScalerT = Point,
    mask_coords: Sequence[int] | None = None,
    func: FuncExceptT | None = None
) -> vs.VideoNode:
    """
    Reduce halo artifacts by nuking everything around edges (and also the edges actually).

    ``rx``, ``ry``, ``darkstr``, ``brightstr``, ``lowsens``, ``highsens``, ``ss`` are all
    configurable per plane and iteration. `tuple` means iteration, `list` plane.

    `rx=(2.0, [2.0, 2.4], [2.2, 2.0, 2.1])` means three iterations.
     * 1st => 2.0 for all planes
     * 2nd => 2.0 for luma, 2.4 for chroma
     * 3rd => 2.2 for luma, 2.0 for u, 2.1 for v

    :param clip:                Source clip.
    :param rx:                  Horizontal radius for halo removal.
    :param ry:                  Vertical radius for halo removal.
    :param darkstr:             Strength factor for dark halos.
    :param brightstr:           Strength factor for bright halos.
    :param lowsens:             Sensitivity setting for defining how weak the dehalo has to be to get fully accepted.
    :param highsens:            Sensitivity setting for define how strong the dehalo has to be to get fully discarded.
    :param sigma_mask:          Blurring strength for the mask.
    :param ss:                  Supersampling factor, to avoid creation of aliasing.
    :param planes:              Planes to process.
    :param show_mask:           Whether to show the computed halo mask.
    :param mask_radius:         Mask expanding radius with ``gradient``.
    :param downscaler:          Scaler used to downscale the clip.
    :param upscaler:            Scaler used to upscale the downscaled clip.
    :param supersampler:        Scaler used to supersampler the rescaled clip to `ss` factor.
    :param supersampler_ref:    Reference scaler used to clamp the supersampled clip. Has to be blurrier.
    :param pre_ss:              Supersampling rate used before anything else.
    :param pre_supersampler:    Supersampler used for ``pre_ss``.
    :param pre_downscaler:      Downscaler used for undoing the upscaling done by ``pre_supersampler``.
    :param func:                Function from where this function was called.

    :return:                    Dehaloed clip.
    """

    func = func or dehalo_alpha

    assert check_variable(clip, func)

    if FieldBased.from_video(clip).is_inter:
        raise UnsupportedFieldBasedError('Only progressive video is supported!', func)

    InvalidColorFamilyError.check(clip, (vs.GRAY, vs.YUV), func)

    planes = normalize_planes(clip, planes)

    downscaler = Scaler.ensure_obj(downscaler, func)
    upscaler = Scaler.ensure_obj(upscaler, func)
    pre_supersampler = Scaler.ensure_obj(pre_supersampler, func)
    pre_downscaler = Scaler.ensure_obj(pre_downscaler, func)

    if ry is None:
        ry = rx

    work_clip, *chroma = split(clip) if planes == [0] else (clip, )

    if pre_ss > 1.0:
        work_clip = pre_supersampler.scale(
            work_clip, mod4(work_clip.width * pre_ss), mod4(work_clip.height * pre_ss)
        )

    def _rescale(clip: vs.VideoNode, rx: float, ry: float) -> vs.VideoNode:
        return upscaler.scale(downscaler.scale(
            clip, mod4(clip.width / rx), mod4(clip.height / ry)
        ), clip.width, clip.height)

    values = _dehalo_schizo_norm(rx, ry, darkstr, brightstr, lowsens, highsens, ss)

    for rx_i, ry_i, darkstr_i, brightstr_i, lowsens_i, highsens_i, ss_i in values:
        if not all(x >= 1 for x in (*ss_i, *rx_i, *ry_i)):
            raise CustomIndexError('ss, rx, and ry must all be bigger than 1.0!', func)

        if not all(0 <= x <= 1 for x in (*brightstr_i, *darkstr_i)):
            raise CustomIndexError('brightstr, darkstr must be between 0.0 and 1.0!', func)

        if not all(0 <= x <= 100 for x in (*lowsens_i, *highsens_i)):
            raise CustomIndexError('lowsens and highsens must be between 0 and 100!', func)

        if len(set(rx_i)) == len(set(ry_i)) == 1 or planes == [0] or work_clip.format.num_planes == 1:  # type: ignore
            dehalo = _rescale(work_clip, rx_i[0], ry_i[0])
        else:
            dehalo = join([_rescale(plane, rxp, ryp) for plane, rxp, ryp in zip(split(work_clip), rx_i, ry_i)])

        mask = _dehalo_mask(work_clip, dehalo, lowsens_i, highsens_i, sigma_mask, mask_radius, mask_coords, planes)

        if show_mask:
            return mask

        dehalo = dehalo.std.MaskedMerge(work_clip, mask, planes)

        dehalo = _dehalo_supersample_minmax(work_clip, dehalo, ss_i, supersampler, supersampler_ref, planes, func)

        work_clip = dehalo = _limit_dehalo(work_clip, dehalo, darkstr_i, brightstr_i, planes)

    if (dehalo.width, dehalo.height) != (clip.width, clip.height):
        dehalo = pre_downscaler.scale(work_clip, clip.width, clip.height)

    if not chroma:
        return dehalo

    return join([dehalo, *chroma], clip.format.color_family)


def dehalo_sigma(
    clip: vs.VideoNode, brightstr: FloatIterArr = 1.0, darkstr: FloatIterArr = 0.0,
    lowsens: FloatIterArr = 50.0, highsens: FloatIterArr = 50.0, ss: FloatIterArr = 1.5,
    blur_func: Prefilter = Prefilter.GAUSS, planes: PlanesT = 0,
    supersampler: ScalerT = Lanczos(3), supersampler_ref: ScalerT = Mitchell,
    pre_ss: float = 1.0, pre_supersampler: ScalerT = Nnedi3(0, field=0, shifter=NoShift),
    pre_downscaler: ScalerT = Point, mask_radius: RadiusT = 1, sigma_mask: float | bool = False,
    mask_coords: Sequence[int] | None = None,
    show_mask: bool = False, func: FuncExceptT | None = None, **kwargs: Any
) -> vs.VideoNode:
    func = func or dehalo_alpha

    assert check_variable(clip, func)

    if FieldBased.from_video(clip).is_inter:
        raise UnsupportedFieldBasedError('Only progressive video is supported!', func)

    InvalidColorFamilyError.check(clip, (vs.GRAY, vs.YUV), func)

    planes = normalize_planes(clip, planes)

    pre_supersampler = Scaler.ensure_obj(pre_supersampler, func)
    pre_downscaler = Scaler.ensure_obj(pre_downscaler, func)

    work_clip, *chroma = split(clip) if planes == [0] else (clip, )

    if pre_ss > 1.0:
        work_clip = pre_supersampler.scale(
            work_clip, mod4(work_clip.width * pre_ss), mod4(work_clip.height * pre_ss)
        )

    values = _dehalo_schizo_norm(darkstr, brightstr, lowsens, highsens, ss)

    for darkstr_i, brightstr_i, lowsens_i, highsens_i, ss_i in values:
        if not all(x >= 1 for x in ss_i):
            raise CustomIndexError('ss must all be bigger than 1.0!', func)

        if not all(0 <= x <= 1 for x in (*brightstr_i, *darkstr_i)):
            raise CustomIndexError('brightstr, darkstr must be between 0.0 and 1.0!', func)

        if not all(0 <= x <= 100 for x in (*lowsens_i, *highsens_i)):
            raise CustomIndexError('lowsens and highsens must be between 0 and 100!', func)

        dehalo = blur_func(work_clip, planes=planes, **kwargs)

        mask = _dehalo_mask(work_clip, dehalo, lowsens_i, highsens_i, sigma_mask, mask_radius, mask_coords, planes)

        if show_mask:
            return mask

        dehalo = dehalo.std.MaskedMerge(work_clip, mask, planes)

        dehalo = _dehalo_supersample_minmax(work_clip, dehalo, ss_i, supersampler, supersampler_ref, planes, func)

        work_clip = dehalo = _limit_dehalo(work_clip, dehalo, darkstr_i, brightstr_i, planes)

    if (dehalo.width, dehalo.height) != (clip.width, clip.height):
        dehalo = pre_downscaler.scale(work_clip, clip.width, clip.height)

    if not chroma:
        return dehalo

    return join([dehalo, *chroma], clip.format.color_family)


def dehalomicron(
    clip: vs.VideoNode, brz: float = 0.075, sigma: float = 1.55, sigma0: float = 1.15, ss: float = 1.65,
    pre_ss: bool = True, dampen: float | list[float] | tuple[float | list[float], bool | None] = 0.65,
    sigma_ref: float = 4.3333, planes: PlanesT = 0, fdehalo_kwargs: KwargsT | None = None, **kwargs: Any
) -> vs.VideoNode:
    func = FunctionUtil(clip, dehalomicron, planes, (vs.GRAY, vs.YUV))

    if FieldBased.from_video(clip).is_inter:
        raise UnsupportedFieldBasedError('Only progressive video is supported!', func.func)

    fdehalo_kwargs = KwargsT(edgeproc=0.5, ss=1.5 if pre_ss else 2.0) | (fdehalo_kwargs or {})

    y = get_y(func.work_clip)

    y_mask = retinex(y)

    dehalo_ref0 = dehalo_sigma(func.work_clip, sigma=sigma0, planes=planes)
    dehalo_ref0mask = dehalo_sigma(y_mask, sigma=sigma0 + sigma0 * 0.09)

    ymask_ref0 = gauss_blur(y_mask, sigma=sigma_ref)

    dehalo_mask = norm_expr([dehalo_ref0mask, y_mask], 'x y - abs 100 *')
    dehalo_mask = RemoveGrainMode.BOX_BLUR_NO_CENTER(dehalo_mask)
    dehalo_mask = RemoveGrainMode.MINMAX_MEDIAN_OPP(dehalo_mask)

    if brz:
        dmask_expr = (
            f"x {scale_mask(abs(brz), 32, y)} "
            f"{'>' if brz < 0.0 else '>'} 0 x 2 * ?"
        )
    else:
        dmask_expr = 'x 2 *'

    dehalo_mask = norm_expr(dehalo_mask, dmask_expr)

    fine_edge_mask = fine_dehalo.mask(norm_expr([y_mask, ymask_ref0], 'y x -'))
    dehalo_mask = norm_expr([dehalo_mask, y_mask, ymask_ref0, fine_edge_mask], 'y z + 2 / x < x and x abs a ?')
    dehalo_mask = RemoveGrainMode.EDGE_CLIP_STRONG(dehalo_mask)

    actual_dehalo = dehalo_sigma(
        func.work_clip, pre_ss=1 + pre_ss, sigma=sigma, ss=ss - 0.5 * pre_ss, planes=func.norm_planes, **kwargs
    )
    dehalo_ref = fine_dehalo(func.work_clip, planes=func.norm_planes, **fdehalo_kwargs)  # type: ignore[arg-type]

    dehalo_min = ExprOp.MIN(actual_dehalo, dehalo_ref, planes=func.norm_planes)

    dehalo = limit_filter(actual_dehalo, func.work_clip, dehalo_ref, planes=func.norm_planes)
    dehalo = dehalo.std.MaskedMerge(dehalo_min, dehalo_mask, func.norm_planes)

    if isinstance(dampen, tuple):
        dampen_amt, dampen_rev = dampen
    else:
        dampen_amt, dampen_rev = dampen, None

    if dampen_rev is None:
        dampen_rev = not pre_ss

    dampen_amt = func.norm_seq(dampen_amt)

    if max(dampen_amt) > 0.0:
        dehalo_ref0 = dehalo.std.Merge(dehalo_ref0, dampen_amt)

    rep = RepairMode.MINMAX_SQUARE_REF_CLOSE if pre_ss else RepairMode.MINMAX_SQUARE_REF1

    clips = (dehalo, dehalo_ref0)

    dehalo = rep(*(reversed(clips) if dampen_rev else clips), func.norm_planes)  # type: ignore

    return func.return_clip(dehalo)


def dehalo_merge(
    clip: vs.VideoNode, dehalo: vs.VideoNode, darkstr: list[float] | float = 0.0, brightstr: list[float] | float = 1.0,
    lowsens: list[float] | float = 50.0, highsens: list[float] | float = 50.0, sigma_mask: float | bool = False,
    ss: list[float] | float = 1.5, planes: PlanesT = 0, show_mask: bool = False, mask_radius: RadiusT = 1,
    supersampler: ScalerT = Lanczos(3), supersampler_ref: ScalerT = Mitchell, pre_ss: float = 1.0,
    pre_supersampler: ScalerT = Nnedi3(0, field=0, shifter=NoShift), pre_downscaler: ScalerT = Point,
    mask_coords: Sequence[int] | None = None, func: FuncExceptT | None = None
) -> vs.VideoNode:
    """
    Merge dehaloed clip onto the source clip.

    ``darkstr``, ``brightstr``, ``lowsens``, ``highsens``, ``ss`` are all configurable per plane.

    :param clip:                Source clip.
    :param dehalo:              Dehaloed clip.
    :param darkstr:             Strength factor for dark halos.
    :param brightstr:           Strength factor for bright halos.
    :param lowsens:             Sensitivity setting for defining how weak the dehalo has to be to get fully accepted.
    :param highsens:            Sensitivity setting for define how strong the dehalo has to be to get fully discarded.
    :param sigma_mask:          Blurring strength for the mask.
    :param ss:                  Supersampling factor, to avoid creation of aliasing.
    :param planes:              Planes to process.
    :param show_mask:           Whether to show the computed halo mask.
    :param mask_radius:         Mask expanding radius with ``gradient``.
    :param supersampler:        Scaler used to supersampler the rescaled clip to `ss` factor.
    :param supersampler_ref:    Reference scaler used to clamp the supersampled clip. Has to be blurrier.
    :param pre_ss:              Supersampling rate used before anything else.
    :param pre_supersampler:    Supersampler used for ``pre_ss``.
    :param pre_downscaler:      Downscaler used for undoing the upscaling done by ``pre_supersampler``.
    :param func:                Function from where this function was called.

    :return:                    Merged clip.
    """

    func = func or dehalo_merge

    assert check_ref_clip(clip, dehalo, func)
    assert check_variable_format(clip, func)

    if FieldBased.from_video(clip).is_inter:
        raise UnsupportedFieldBasedError('Only progressive video is supported!', func)

    InvalidColorFamilyError.check(clip, (vs.GRAY, vs.YUV), func)

    planes = normalize_planes(clip, planes)

    pre_supersampler = Scaler.ensure_obj(pre_supersampler, func)
    pre_downscaler = Scaler.ensure_obj(pre_downscaler, func)

    work_clip, *chroma = split(clip) if planes == [0] else (clip, )
    dehalo = split(dehalo)[0] if planes == [0] else dehalo

    if pre_ss > 1.0:
        work_clip = pre_supersampler.scale(
            work_clip, mod4(work_clip.width * pre_ss), mod4(work_clip.height * pre_ss)
        )

    darkstr_i, brightstr_i, lowsens_i, highsens_i, ss_i = next(
        _dehalo_schizo_norm(darkstr, brightstr, lowsens, highsens, ss)  # type: ignore[call-overload]
    )

    if not all(x >= 1 for x in ss_i):
        raise CustomIndexError('ss must be bigger than 1.0!', func)

    if not all(0 <= x <= 1 for x in (*brightstr_i, *darkstr_i)):
        raise CustomIndexError('brightstr, darkstr must be between 0.0 and 1.0!', func)

    if not all(0 <= x <= 100 for x in (*lowsens_i, *highsens_i)):
        raise CustomIndexError('lowsens and highsens must be between 0 and 100!', func)

    mask = _dehalo_mask(work_clip, dehalo, lowsens_i, highsens_i, sigma_mask, mask_radius, mask_coords, planes)

    if show_mask:
        return mask

    dehalo = dehalo.std.MaskedMerge(work_clip, mask, planes)

    dehalo = _dehalo_supersample_minmax(work_clip, dehalo, ss_i, supersampler, supersampler_ref, planes, func)

    work_clip = dehalo = _limit_dehalo(work_clip, dehalo, darkstr_i, brightstr_i, planes)

    if (dehalo.width, dehalo.height) != (clip.width, clip.height):
        dehalo = pre_downscaler.scale(work_clip, clip.width, clip.height)

    if not chroma:
        return dehalo

    return join([dehalo, *chroma], clip.format.color_family)
