from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import vapoursynth as vs

from ..abstract import _Antialiaser, _FullInterpolate, Antialiaser, DoubleRater, SingleRater, SuperSampler

__all__ = ['Eedi2', 'Eedi2SS', 'Eedi2SR', 'Eedi2DR']

core = vs.core


@dataclass
class EEDI2(_FullInterpolate, _Antialiaser):
    mthresh: int = 10
    lthresh: int = 20
    vthresh: int = 20
    estr: int = 2
    dstr: int = 4
    maxd: int = 24
    pp: int = 1

    cuda: bool = dc_field(default=False, kw_only=True)

    def _full_interpolate_enabled(self, x: bool, y: bool) -> bool:
        return self.cuda and x and y

    def get_aa_args(self, clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
        return dict(
            mthresh=self.mthresh, lthresh=self.lthresh, vthresh=self.vthresh,
            estr=self.estr, dstr=self.dstr, maxd=self.maxd, pp=self.pp
        )

    def _interpolate(self, clip: vs.VideoNode, double_y: bool, **kwargs: Any) -> vs.VideoNode:
        if self.cuda:
            clip = core.eedi2cuda.EEDI2(clip, self.field, **kwargs)
        else:
            clip = core.eedi2.EEDI2(clip, self.field, **kwargs)

        if double_y:
            return clip

        clip = clip.std.SeparateFields(not self.field)[::2]

        return self.shifter.shift(clip, (0.5 - 0.75 * self.field, 0))

    def _full_interpolate(self, clip: vs.VideoNode, double_y: bool, double_x: bool, **kwargs: Any) -> vs.VideoNode:
        return core.eedi2cuda.Enlarge2(clip, **kwargs)

    _shift = -0.5


class Eedi2SS(EEDI2, SuperSampler):
    ...


class Eedi2SR(EEDI2, SingleRater):
    ...


class Eedi2DR(EEDI2, DoubleRater):
    ...


class Eedi2(EEDI2, Antialiaser):
    ...
