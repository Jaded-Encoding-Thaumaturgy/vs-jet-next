from __future__ import annotations

from vsexprtools import complexpr_available, norm_expr
from vstools import NotFoundEnumValue, PlanesT, check_variable, core, pick_func_stype, vs

from .aka_expr import removegrain_aka_exprs, repair_aka_exprs
from .enum import (
    BlurMatrix, RemoveGrainMode, RemoveGrainModeT, RepairMode, RepairModeT, VerticalCleanerMode, VerticalCleanerModeT
)

__all__ = [
    'repair', 'removegrain',
    'clense', 'backward_clense', 'forward_clense',
    'vertical_cleaner'
]


def repair(clip: vs.VideoNode, repairclip: vs.VideoNode, mode: RepairModeT) -> vs.VideoNode:
    assert check_variable(clip, repair)
    assert check_variable(repairclip, repair)

    if mode not in RepairMode:
        raise NotFoundEnumValue('Invalid RepairMode specified!', repair, mode)
    
    if mode == RepairMode.NONE:
        return clip

    if not complexpr_available:
        if mode in (RepairMode.CLIP_REF_RG20, RepairMode.CLIP_REF_RG23) and clip.format.sample_type == vs.FLOAT:
            raise NotFoundEnumValue('Specified RepairMode for rgsf is not implemented!', repair, mode)

        return pick_func_stype(clip, core.rgvs.Repair, core.rgsf.Repair)(clip, repairclip, mode)

    return norm_expr([clip, repairclip], repair_aka_exprs[mode])


def removegrain(clip: vs.VideoNode, mode: RemoveGrainModeT) -> vs.VideoNode:
    assert check_variable(clip, removegrain)

    if mode not in RemoveGrainMode:
        raise NotFoundEnumValue('Invalid RemoveGrainMode specified!', removegrain, mode)
    
    if mode == RemoveGrainMode.NONE:
        return clip

    if clip.format.sample_type == vs.INTEGER and mode in range(1, 24 + 1):
        if hasattr(core, "zsmooth"):
            return clip.zsmooth.RemoveGrain(mode)

        if hasattr(core, 'rgvs'):
            return clip.rgvs.RemoveGrain(mode)

    if not complexpr_available:
        return clip.zsmooth.RemoveGrain(mode)
    
    if RemoveGrainMode.BOB_TOP_CLOSE <= mode <= RemoveGrainMode.BOB_BOTTOM_INTER:
        return pick_func_stype(clip, core.lazy.rgvs.RemoveGrain, core.lazy.zsmooth.RemoveGrain)(clip, mode)

    match mode:
        case RemoveGrainMode.MINMAX_MEDIAN:
            return clip.std.Median()

        case RemoveGrainMode.BINOMIAL_BLUR:
            return BlurMatrix.BINOMIAL()(clip)

        case RemoveGrainMode.BOX_BLUR_NO_CENTER:
            return BlurMatrix.MEAN_NO_CENTER()(clip)

        case RemoveGrainMode.BOX_BLUR:
            return BlurMatrix.MEAN()(clip)

        case _:
            return norm_expr(clip, removegrain_aka_exprs[mode])


def clense(
    clip: vs.VideoNode,
    previous_clip: vs.VideoNode | None = None, next_clip: vs.VideoNode | None = None,
    planes: PlanesT = None
) -> vs.VideoNode:
    return pick_func_stype(clip, core.lazy.rgvs.Clense, core.lazy.rgsf.Clense)(clip, previous_clip, next_clip, planes)


def forward_clense(clip: vs.VideoNode, planes: PlanesT = None) -> vs.VideoNode:
    return pick_func_stype(clip, core.lazy.rgvs.ForwardClense, core.lazy.rgsf.ForwardClense)(clip, planes)


def backward_clense(clip: vs.VideoNode, planes: PlanesT = None) -> vs.VideoNode:
    return pick_func_stype(clip, core.lazy.rgvs.BackwardClense, core.lazy.rgsf.BackwardClense)(clip, planes)


def vertical_cleaner(clip: vs.VideoNode, mode: VerticalCleanerModeT = VerticalCleanerMode.MEDIAN) -> vs.VideoNode:
    if mode not in VerticalCleanerMode:
        raise NotFoundEnumValue('Invalid VerticalCleanerMode specified!', vertical_cleaner, mode)
    
    if mode == VerticalCleanerMode.NONE:
        return clip

    return pick_func_stype(clip, core.lazy.rgvs.VerticalCleaner, core.lazy.rgsf.VerticalCleaner)(clip, mode)
