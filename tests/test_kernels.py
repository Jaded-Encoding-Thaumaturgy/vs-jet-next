from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vskernels import Bicubic, BicubicAuto, Bilinear, Lanczos, Point, Spline16, Spline36, Spline64, ZimgComplexKernel
from vstools import core, vs


clip = core.std.BlankClip(format=vs.YUV420P16, width=1920, height=1080)
clip_descaled = core.std.BlankClip(format=vs.YUV420P16, width=1600, height=900)
kernels = [Bicubic, BicubicAuto, Bilinear, Lanczos, Point, Spline16, Spline36, Spline64]

@pytest.mark.parametrize("kernel", kernels)
def test_blur(kernel: type[ZimgComplexKernel]) -> None:
    with patch.object(kernel, 'scale_function') as mock_scale_function:
        kernel.scale(clip, 1600, 900, blur=1.15)

        mock_scale_function.assert_called_once()
        assert mock_scale_function.call_args.kwargs["blur"] == 1.15

    with patch.object(kernel, 'descale_function', new=MagicMock(return_value=clip_descaled)) as mock_descale_function:
        kernel.descale(clip, 1600, 900, blur=1.15)

        mock_descale_function.assert_called_once()
        assert mock_descale_function.call_args.kwargs["blur"] == 1.15
