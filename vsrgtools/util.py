from __future__ import annotations

from typing import Any

from vstools import ConvMode, GenericVSFunction, KwargsT, Nb, check_variable_format, join, normalize_seq, plane, vs

from .enum import BlurMatrix

__all__ = [
    'wmean_matrix', 'mean_matrix',
    'normalize_radius'
]

wmean_matrix = list(BlurMatrix.BINOMIAL(1, mode=ConvMode.SQUARE))
mean_matrix = list(BlurMatrix.MEAN(1, mode=ConvMode.SQUARE))


def normalize_radius(
    clip: vs.VideoNode, func: GenericVSFunction, radius: list[Nb] | tuple[str, list[Nb]],
    planes: list[int], **kwargs: Any
) -> vs.VideoNode:
    assert check_variable_format(clip, normalize_radius)

    name, radius = radius if isinstance(radius, tuple) else ('radius', radius)

    radius = normalize_seq(radius, clip.format.num_planes)

    def _get_kwargs(rad: Nb) -> KwargsT:
        return kwargs | {name: rad, 'planes': planes}

    if len(set(radius)) > 0:
        if len(planes) != 1:
            return join([
                func(plane(clip, i), **_get_kwargs(rad)) for i, rad in enumerate(radius)
            ])

        radius_i = radius[planes[0]]
    else:
        radius_i = radius[0]

    return func(clip, **_get_kwargs(radius_i))
