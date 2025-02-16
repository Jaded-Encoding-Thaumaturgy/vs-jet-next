from __future__ import annotations

from functools import partial
from inspect import Signature
from typing import Callable

from vsexprtools import complexpr_available, norm_expr
from vstools import (
    CustomValueError, GenericVSFunction, PlanesT, check_ref_clip, check_variable, clamp_arr, get_neutral_value,
    iterate, normalize_planes, to_arr, vs, core
)

from .blur import box_blur, median_blur, min_blur
from .enum import BlurMatrix, RemoveGrainMode, RemoveGrainModeT, RepairMode, RepairModeT
from .rgtools import removegrain, repair

__all__ = [
    'contrasharpening', 'contra',
    'contrasharpening_dehalo', 'contra_dehalo',
    'contrasharpening_median', 'contra_median',
    'fine_contra'
]


def contrasharpening(
    flt: vs.VideoNode, src: vs.VideoNode, radius: int | list[int] = 1,
    sharp: vs.VideoNode | GenericVSFunction | None = None,
    mode: RepairModeT = RepairMode.MINMAX_SQUARE3, planes: PlanesT = 0
) -> vs.VideoNode:
    """
    contra-sharpening: sharpen the denoised clip, but don't add more to any pixel than what was previously removed.
    Script by Didée, at the VERY GRAINY thread (http://forum.doom9.org/showthread.php?p=1076491#post1076491)

    :param flt:         Filtered clip
    :param src:         Source clip
    :param radius:      Spatial radius for sharpening.
    :param sharp:       Optional pre-sharpened clip or function to use.
    :param mode:        Mode of rgvs.Repair to limit the difference
    :param planes:      Planes to process, defaults to None

    :return:            Contrasharpened clip
    """

    assert check_variable(src, contrasharpening)
    assert check_variable(flt, contrasharpening)
    check_ref_clip(src, flt, contrasharpening)

    planes = normalize_planes(flt, planes)

    # Damp down remaining spots of the denoised clip
    if isinstance(sharp, vs.VideoNode):
        sharpened = sharp
    elif callable(sharp):
        sharpened = sharp(flt)
    else:
        damp = min_blur(flt, radius, planes=planes)
        blurred = BlurMatrix.BINOMIAL(radius=radius)(damp, planes=planes)

    # Difference of a simple kernel blur
    diff_blur = core.std.MakeDiff(
        sharpened if sharp else damp,
        flt if sharp else blurred,
        planes
    )

    # Difference achieved by the filtering
    diff_flt = src.std.MakeDiff(flt, planes)

    # Limit the difference to the max of what the filtering removed locally
    limit = repair(diff_blur, diff_flt, mode)

    # abs(diff) after limiting may not be bigger than before
    # Apply the limited difference (sharpening is just inverse blurring)
    expr = 'x neutral - X! y neutral - Y! X@ abs Y@ abs < X@ Y@ ? z +'

    return norm_expr([limit, diff_blur, flt], expr, planes)


def contrasharpening_dehalo(
    flt: vs.VideoNode, src: vs.VideoNode, level: float = 1.4, alpha: float = 2.49, planes: PlanesT = 0
) -> vs.VideoNode:
    """
    :param dehaloed:    Dehaloed clip
    :param src:         Source clip
    :param level:       Strength level
    :return:            Contrasharpened clip
    """
    assert check_variable(src, contrasharpening)
    assert check_variable(flt, contrasharpening)
    check_ref_clip(src, flt, contrasharpening)

    planes = normalize_planes(flt, planes)

    blur = BlurMatrix.BINOMIAL()(flt, planes)
    blur2 = median_blur(blur, 2, planes=planes)
    blur2 = iterate(blur2, partial(repair, repairclip=blur), 2, mode=RepairMode.MINMAX_SQUARE1)

    return norm_expr(
        [blur, blur2, src, flt],
        'x y - {alpha} * {level} * D1! z a - D2! D1@ D2@ xor 0 D1@ abs D2@ abs < D1@ D2@ ? ? a +',
        planes, alpha=alpha, level=level
    )


def contrasharpening_median(
    flt: vs.VideoNode, src: vs.VideoNode,
    mode: RemoveGrainModeT | Callable[..., vs.VideoNode] = box_blur,
    planes: PlanesT = 0
) -> vs.VideoNode:
    """
    :param flt:         Filtered clip
    :param src:         Source clip
    :param mode:        Function or the RemoveGrain mode used to blur/repair the filtered clip.
    :param planes:      Planes to process, defaults to None
    :return:            Contrasharpened clip
    """
    assert check_variable(src, contrasharpening)
    assert check_variable(flt, contrasharpening)
    check_ref_clip(src, flt, contrasharpening)

    planes = normalize_planes(flt, planes)

    if isinstance(mode, (int, list, RemoveGrainMode)):
        repaired = removegrain(flt, mode)
    elif callable(mode):
        repaired = mode(flt, planes=planes)
    else:
        raise CustomValueError('Invalid mode or function passed!', contrasharpening_median)

    if complexpr_available:
        expr = 'x dup + z - D! x y < D@ x y clamp D@ y x clamp ?'
    else:
        expr = 'x dup + z - x y min max x y max min'

    return norm_expr([flt, src, repaired], expr, planes)


def fine_contra(
    flt: vs.VideoNode, src: vs.VideoNode, sharp: float | list[float] | range = 0.75,
    radius: int | list[int] = 1, merge_func: GenericVSFunction | None = None,
    mode: RepairModeT = RepairMode.MINMAX_SQUARE_REF3, planes: PlanesT = 0
) -> vs.VideoNode:
    """
    :param flt:         Filtered clip.
    :param src:         Source clip.
    :param sharp:       Contrast Adaptive Sharpening's sharpening strength.
                        If it's a list, depending on ``merge_func`` being ``None``,
                        it will iterate over with different strengths or merge all with ``merge_func``.
    :param radius:      Spatial radius for contra-sharpening (1-3). Default is 2 for HD / 1 for SD.
    :param merge_func:  Depending on ``sharp``, this will get all sharpened clips and merge them.
    :param mode:        Mode of rgvs.Repair to limit the difference.
    :param planes:      Planes to process, defaults to None.
    :return:            Contrasharpened clip.
    """

    assert check_variable(src, contrasharpening)
    assert check_variable(flt, contrasharpening)
    check_ref_clip(src, flt, contrasharpening)

    neutral = get_neutral_value(flt)

    planes = normalize_planes(flt, planes)

    mblur = min_blur(flt, radius, planes=planes)

    sharp = [1.0 / x for x in sharp if x] if isinstance(sharp, range) else to_arr(sharp)
    sharp = clamp_arr(sharp, 0.0, 1.0)

    if merge_func is None:
        for s in sharp:
            mblur = mblur.cas.CAS(s, planes)
    else:
        mblurs = [mblur.cas.CAS(s, planes) for s in sharp]

        got_p = 'planes' in Signature.from_callable(merge_func).parameters.keys()

        try:
            if got_p:
                mblur = merge_func(*mblurs, planes=planes)
            else:
                mblur = merge_func(*mblurs)
        except Exception:
            if got_p:
                mblur = merge_func(mblurs, planes=planes)
            else:
                mblur = merge_func(mblurs)

    limit = repair(mblur, src.std.MakeDiff(flt, planes), mode)

    if complexpr_available:
        expr = 'x {mid} - LD! y {mid} - BD! LD@ abs BD@ abs < LD@ BD@ ? z +'
    else:
        expr = 'x {mid} - abs y {mid} - abs < x y ? {mid} - z +'

    return norm_expr([limit, mblur, flt], expr, planes, mid=neutral)


contra = contrasharpening
contra_dehalo = contrasharpening_dehalo
contra_median = contrasharpening_median
