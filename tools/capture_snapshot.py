"""D435 스냅샷 저장 — 로봇 없이 M1·M2를 개발하기 위한 데이터 (컬러 + 컬러 정렬 깊이 + 내부 파라미터).

현장과 같은 0.6~0.7m 고정 거치에서 사과를 놓고 찍는다. Enter 를 누를 때마다 1장 저장한다.
저장한 .npz 는 공유 드라이브에 올리고, 코드에서는 aple.hal.replay 로 읽는다.
Piper Studio가 D435를 열고 있으면 실행되지 않으므로 카메라 데몬을 멈춘 뒤 실행한다.

    python tools/capture_snapshot.py --out data/snapshots/1004_desk --note "형광등, 사과 5개"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyrealsense2 as rs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    pipe = rs.pipeline()
    profile = pipe.start(cfg)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    serial = profile.get_device().get_info(rs.camera_info.serial_number)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    align = rs.align(rs.stream.color)

    meta = {"serial": serial, "width": args.width, "height": args.height,
            "intrinsics": [intr.fx, intr.fy, intr.ppx, intr.ppy], "note": args.note}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        for _ in range(30):                      # 자동 노출 안정화 대기
            pipe.wait_for_frames()
        n = len(list(out.glob("*.npz")))
        while True:
            if input(f"[{n}] Enter = 저장, q = 종료 > ").strip().lower() == "q":
                break
            frames = align.process(pipe.wait_for_frames())
            color = np.asanyarray(frames.get_color_frame().get_data()).copy()
            depth_m = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * scale
            path = out / f"snap_{n:04d}.npz"
            np.savez_compressed(path, color=color, depth_m=depth_m,
                                intrinsics=np.array(meta["intrinsics"]), stamp_s=time.time())
            print(f"  저장: {path}  (깊이 유효 {(depth_m > 0).mean() * 100:.0f}%)")
            n += 1
    finally:
        pipe.stop()


if __name__ == "__main__":
    main()
