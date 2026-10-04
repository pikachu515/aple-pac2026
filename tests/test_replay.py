import numpy as np

from aple.hal.replay import iter_snapshots
from aple.schema import Source


def test_saved_snapshot_round_trips(tmp_path):
    color = np.zeros((48, 64, 3), np.uint8)
    depth = np.full((48, 64), 0.65, np.float32)
    for i in range(2):
        np.savez_compressed(tmp_path / f"snap_{i:04d}.npz", color=color, depth_m=depth,
                            intrinsics=np.array([600.0, 600.0, 32.0, 24.0]), stamp_s=1.0 + i)

    snaps = list(iter_snapshots(tmp_path))
    assert [s.scene_id for s in snaps] == [1, 2]
    assert snaps[0].source == Source.TOP
    assert snaps[0].color.shape == (48, 64, 3)
    assert snaps[1].intrinsics == (600.0, 600.0, 32.0, 24.0)
