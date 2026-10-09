"""깊이 거리 확인·보정 — 회색 카드 크기로 실제 거리를 재서 D435 깊이와 비교한다. 역할 A.

#29 '직경이 작게 나온다'의 원인은 깊이였다: 10/7 실물 D435i 는 0.74 m 에서 깊이가 45% 짧았다
(aple/hal/depth_correction.py). 깊이가 틀리면 사과 직경·높이·3D 위치·카메라–로봇 보정이 모두 틀린다.
카메라를 설치하면 캘리브레이션(touch_calib) 전에 이것부터 확인한다.

    python tools/depth_check.py data/snapshots/1007_home2              # 저장 스냅샷 (폴더 또는 .npz 여러 개)
    python tools/depth_check.py --live                                 # Piper Studio 카메라 1장
    python tools/depth_check.py --live --save data/depth_check/0.65m   # 찍은 장면을 저장 (나중에 fit 에 다시 쓴다)
    python tools/depth_check.py fit data/depth_check/0.45m data/depth_check/0.65m   # 보정값 계산

준비: 회색 카드를 트레이 높이에 눕혀 카메라 정면에 둔다. 카드 크기는 자로 재서 live_camera.depth_check.card_mm.
fit 은 카드 높이를 두 가지 이상(예: 트레이 위, 상자 하나 위)으로 찍으면 배율 + 오프셋, 한 가지면 배율만 맞춘다.
표시: '깊이' = 카메라 값, '카드' = 카드 크기로 잰 실제 거리, '보정 후' = 지금 설정의 depth_correction 적용.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from aple.config import load_config  # noqa: E402
from aple.hal.depth_correction import DepthCorrection, card_distance, fit_correction  # noqa: E402
from aple.hal.replay import load_snapshot  # noqa: E402


def snapshot_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        files += sorted(p.glob("*.npz")) if p.is_dir() else [p]
    if not files:
        raise SystemExit("스냅샷(.npz)이 없다")
    return files


def measure(snaps, card_mm, corr: DepthCorrection, max_error: float) -> list:
    rows = []
    print(f"{'장면':32s} {'깊이':>7s} {'카드':>7s} {'오차':>7s} {'보정 후':>8s}   카드 px")
    for name, snap in snaps:
        cd = card_distance(snap, card_mm)
        if cd is None:
            print(f"{name:32s} 회색 카드를 못 찾았다 (눕혀서 화면 안에, 깊이가 잡히게)")
            continue
        after = float(corr.apply(np.array([cd.z_depth_m], np.float32))[0]) / cd.z_true_m - 1
        flag = "" if abs(after) <= max_error else "  ← 허용 초과"
        print(f"{name:32s} {cd.z_depth_m:7.3f} {cd.z_true_m:7.3f} {cd.error * 100:+6.1f}% {after * 100:+7.1f}%"
              f"   {cd.size_px[0]:.0f}×{cd.size_px[1]:.0f}{flag}")
        rows.append(cd)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="[fit] 스냅샷 폴더 또는 .npz (맨 앞에 fit 을 쓰면 보정값 계산)")
    ap.add_argument("--live", action="store_true", help="Piper Studio 공유 메모리에서 1장")
    ap.add_argument("--save", type=Path, help="--live 장면을 이 폴더에 저장 (보정 전 깊이)")
    ap.add_argument("--card-mm", help="카드 긴 변,짧은 변 (기본 live_camera.depth_check.card_mm)")
    args = ap.parse_args(argv)
    args.cmd = "fit" if args.paths[:1] == ["fit"] else "check"
    args.paths = [Path(p) for p in args.paths if p != "fit" or args.cmd != "fit"]

    cfg = load_config()
    dc = cfg["live_camera"].get("depth_check", {})
    card_mm = tuple(float(v) for v in (args.card_mm.split(",") if args.card_mm else dc.get("card_mm", (130, 100))))
    max_error = float(dc.get("max_error", 0.02))
    corr = DepthCorrection.from_cfg(cfg)

    snaps = []
    if args.live:
        from aple.hal.live_snapshot import PiperLiveCamera
        snap = PiperLiveCamera.from_config(cfg)()
        snaps.append(("live", snap))
        if args.save:
            args.save.mkdir(parents=True, exist_ok=True)
            out = args.save / f"snap_{len(list(args.save.glob('snap_*.npz'))):04d}.npz"
            np.savez_compressed(out, color=snap.color, depth_m=snap.depth_m, intrinsics=np.array(snap.intrinsics),
                                stamp_s=time.time())
            print(f"저장: {out}")
    if args.paths:
        snaps += [(f"{f.parent.name}/{f.name}", load_snapshot(f)) for f in snapshot_files(args.paths)]
    if not snaps:
        ap.error("스냅샷 경로 또는 --live")

    print(f"카드 {card_mm[0]:g}×{card_mm[1]:g} mm, 현재 depth_correction gain {corr.gain:g}, "
          f"offset_per_m {corr.offset_per_m:g}")
    rows = measure(snaps, card_mm, corr, max_error)
    if not rows:
        return 1
    if args.cmd == "check":
        worst = max(abs(float(corr.apply(np.array([r.z_depth_m], np.float32))[0]) / r.z_true_m - 1) for r in rows)
        if worst <= max_error:
            print(f"통과 — 보정 후 오차 최대 {worst * 100:.1f}% (허용 {max_error * 100:.0f}%)")
            return 0
        print(f"불합격 — 보정 후 오차 최대 {worst * 100:.1f}%. 카메라 재보정(On-Chip + Tare) 후 다시 확인하거나, "
              f"카드 높이 2가지 이상으로 찍어 'fit' 한다")
        return 2

    fit = fit_correction([r.z_depth_m for r in rows], [r.z_true_m for r in rows])
    zm = np.array([r.z_depth_m for r in rows], np.float32)
    resid = fit.apply(zm) / np.array([r.z_true_m for r in rows]) - 1
    kind = "배율만 (카드 거리가 한 가지)" if fit.offset_per_m == 0 else "배율 + 오프셋"
    print(f"\nfit: gain {fit.gain:.4f}, offset_per_m {fit.offset_per_m:.4f}  [{kind}], 맞춘 뒤 오차 최대 "
          f"{np.max(np.abs(resid)) * 100:.1f}%")
    print("설정에 넣기:")
    print(f"  python tools/field_values.py set live_camera.depth_correction.gain {fit.gain:.4f} --note \"depth_check fit\"")
    print(f"  python tools/field_values.py set live_camera.depth_correction.offset_per_m {fit.offset_per_m:.4f} "
          f"--note \"depth_check fit\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
