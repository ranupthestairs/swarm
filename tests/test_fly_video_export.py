from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swarm.core.fly_viewer import export_video


def test_export_video_writes_playable_h264(tmp_path: Path) -> None:
    pytest.importorskip("imageio_ffmpeg")
    frames = []
    for idx in range(6):
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        frame[:, :, 0] = idx * 40
        frames.append(frame)

    output = tmp_path / "sample.mp4"
    export_video(frames, output, fps=25)

    assert output.is_file()
    assert output.stat().st_size > 0

    import imageio.v3 as iio

    meta = iio.immeta(output)
    assert meta.get("codec") == "h264"
    assert meta.get("pix_fmt", "").startswith("yuv420p")
    frame = iio.imread(output, index=0)
    assert frame.shape == (540, 960, 3)
