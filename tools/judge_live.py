"""Piper Studio 를 켠 채로 D435 영상을 받아 실시간 착색 판정 (역할 C · 대회 연결 경로 검증).

capture_snapshot.py 와 달리 카메라를 직접 열지 않는다. Piper Studio(rsd)가 공유 메모리로 내보내는
컬러·깊이를 읽어 깊이를 디코딩·정렬하고(aple/hal/depth_align.py) M2 판정(inspect_scene)을 돌린다.
대회에서 판정 모듈이 받게 될 것과 같은 입력이다.

준비
  1. Piper Studio 에서 D435 를 켜고, 회색 카드 보정 → 카메라 프로파일 저장
  2. 깊이→컬러 보정값: python tools/rs_check.py --save-calib configs/calib/d435.json (장치당 한 번, rsd 를 잠시 멈추고)
  3. 깊이 인코딩 창(near/far)을 Piper Studio 카메라 설정에서 확인해 --near --far 로 넘긴다 (기본 150/1200)

    python tools/judge_live.py                               # 창에 판정 결과 표시
    python tools/judge_live.py --card-roi 20,20,80,60        # 시야 안 회색 카드로 남은 색 치우침 보정
    python tools/judge_live.py --once --out data/live         # 한 장만 판정하고 결과 저장

창 키:  s 현재 장면 저장(스냅샷 .npz + 오버레이)   q 종료
저장한 .npz 는 color_session.py analyze / truth 로 그대로 다시 분석할 수 있다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aple.config import load_config  # noqa: E402
from aple.hal.depth_align import D435Calib, DepthEncoding, snapshot_from_piper  # noqa: E402
from aple.hal.shm_frames import SHM_DIR, ShmError, ShmReader, find_realsense_serials, read_pair, segment_path  # noqa: E402
from aple.inspection.color import apply_gain, calibrate_card, lighting_status  # noqa: E402
from aple.inspection.coloration import measure_view, params_from_config  # noqa: E402
from aple.inspection.pipeline import inspect_scene  # noqa: E402
from aple.inspection.standalone import segment_apples  # noqa: E402
from aple.schema import Grade, InspectionPolicy  # noqa: E402
from tools.color_session import GRADE_EN, overlay  # noqa: E402

GRADE_KO = {Grade.UPPER: "상", Grade.MIDDLE: "중", Grade.REJECT: "리젝트"}


def judge_frame(color_raw, depth_enc, enc, calib, p, scene_id, card_calib=None,
                policy=InspectionPolicy.NEVER):
    """→ (오버레이 이미지, 결과 행 목록, 정렬된 스냅샷)."""
    snap = snapshot_from_piper(color_raw, depth_enc, enc, calib, scene_id, time.time())
    light = lighting_status(color_raw, card_calib) if card_calib else None
    snap.color = apply_gain(color_raw, card_calib)
    dets = segment_apples(snap.color, snap.depth_m, snap.intrinsics, scene_id=scene_id)
    grades = {g.apple_id: g for g in inspect_scene(snap, dets, None, p, light=light, policy=policy)}
    items, rows = [], []
    for d in dets:
        g = grades[d.det_id]
        vm = measure_view(snap.color, snap.depth_m, d.mask, snap.intrinsics, p)
        items.append((d.det_id, vm, f"{g.ratio * 100:.0f}% {GRADE_EN[g.grade]}" if vm else "?"))
        rows.append({"apple": d.det_id, "ratio": round(g.ratio, 3), "grade": GRADE_KO[g.grade],
                     "reason": g.reject_reason.name, "final": g.final,
                     "diameter_mm": round(vm.sphere.radius * 2000, 1) if vm else None,
                     "light_dE": round(light.delta_e, 2) if light else None})
    return overlay(snap.color, items), rows, snap


def save(out: Path, n: int, color_raw, snap, image) -> None:
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"snap_{n:04d}.npz", color=color_raw, depth_m=snap.depth_m,
                        intrinsics=np.array(snap.intrinsics), stamp_s=snap.stamp_s)
    cv2.imwrite(str(out / f"live_{n:04d}_overlay.png"), image)


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serial", help="D435 시리얼 (기본: 공유 메모리에서 자동 탐색)")
    ap.add_argument("--calib", type=Path, default=ROOT / "configs" / "calib" / "d435.json")
    ap.add_argument("--near", type=float, default=150.0, help="Piper Studio 깊이 인코딩 near_mm")
    ap.add_argument("--far", type=float, default=1200.0, help="Piper Studio 깊이 인코딩 far_mm")
    ap.add_argument("--card-roi", help="x,y,w,h — 회색 카드 영역 (첫 장면에서 남은 색 치우침 보정값을 만든다)")
    ap.add_argument("--variety", choices=["fuji", "hongro"])
    ap.add_argument("--policy", choices=[p.name for p in InspectionPolicy], default="NEVER")
    ap.add_argument("--once", action="store_true", help="한 장만 판정하고 끝낸다")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "live")
    ap.add_argument("--shm-dir", type=Path, default=SHM_DIR)
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args(argv)

    if not args.calib.exists():
        raise SystemExit(f"깊이→컬러 보정 파일이 없다: {args.calib}\n"
                         "  → systemctl --user stop piper-rsd && python tools/rs_check.py --save-calib "
                         f"{args.calib} && systemctl --user start piper-rsd")
    calib = D435Calib.load(args.calib)
    serial = args.serial or next(iter(find_realsense_serials(args.shm_dir)), None)
    if serial is None:
        raise SystemExit("공유 메모리에 RealSense 컬러 영상이 없다 — Piper Studio 에서 D435 를 켰는지 확인")
    try:
        color_r = ShmReader(segment_path(f"rs:{serial}:color", args.shm_dir))
        depth_r = ShmReader(segment_path(f"rs:{serial}:depth", args.shm_dir))
    except ShmError as e:
        raise SystemExit(str(e))
    if color_r.shape[:2] != (calib.color.height, calib.color.width):
        print(f"[경고] 컬러 해상도 {color_r.shape[1]}x{color_r.shape[0]} ≠ 보정 파일 "
              f"{calib.color.width}x{calib.color.height} — 해상도를 바꿨다면 --save-calib 을 다시 한다")

    enc = DepthEncoding(args.near, args.far)
    p = params_from_config(load_config(), args.variety)
    policy = InspectionPolicy[args.policy]
    card = None
    print(f"D435 {serial}, 깊이 창 {args.near:.0f}~{args.far:.0f} mm, 품종 {p.variety}, 정책 {policy.name}")

    scene_id, n_saved, rows = 0, len(list(args.out.glob("snap_*.npz"))) if args.out.exists() else 0, []
    shown = False
    try:
        while True:
            pair = read_pair(color_r, depth_r)
            if pair is None:
                print("[대기] 컬러·깊이 프레임을 기다리는 중 (스트림이 켜져 있는지 확인)")
                if args.once:
                    raise SystemExit(1)
                continue
            color_raw, depth_enc, _ = pair
            scene_id += 1
            if card is None and args.card_roi:
                card = calibrate_card(color_raw, tuple(int(v) for v in args.card_roi.split(",")))
                print(f"회색 카드 잔여 보정 게인 R {card.rgb_gain[0]:.2f} G {card.rgb_gain[1]:.2f} B {card.rgb_gain[2]:.2f}")
            image, rows, snap = judge_frame(color_raw, depth_enc, enc, calib, p, scene_id, card, policy)

            if args.once:
                save(args.out, n_saved, color_raw, snap, image)
                for r in rows:
                    print(f"#{r['apple']}  {r['ratio'] * 100:5.0f}%  {r['grade']:<4} {r['reason']}  직경 {r['diameter_mm']} mm")
                print(f"저장: {args.out}")
                return rows
            if args.no_window:
                print(" | ".join(f"#{r['apple']} {r['ratio'] * 100:.0f}% {r['grade']}" for r in rows) or "사과 없음")
                time.sleep(0.5)
                continue
            cv2.imshow("judge_live (s save, q quit)", image)
            shown = True
            k = cv2.waitKey(30) & 0xFF
            if k == ord("q"):
                break
            if k == ord("s"):
                save(args.out, n_saved, color_raw, snap, image)
                print(f"저장 #{n_saved}: " + ", ".join(f"#{r['apple']} {r['grade']}" for r in rows))
                n_saved += 1
    finally:
        color_r.close()
        depth_r.close()
        if shown:                       # 화면 없는 환경(headless OpenCV)에서는 창 함수를 부르지 않는다
            cv2.destroyAllWindows()
    return rows


if __name__ == "__main__":
    main()
