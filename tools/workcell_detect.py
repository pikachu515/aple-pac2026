"""상자·리젝트 종이 구역·사과 집는 영역을 카메라로 찾아 보여 준다 (aple/perception/workcell_detect.py, 역할 A).

실행할 때는 workcell.auto_detect.enabled 로 장면마다 자동으로 찾는다. 이 도구는 현장 점검용이다:
찾은 위치를 base_link 좌표로 출력하고, 영상 위에 그린 그림을 저장한다. --save 는 찾은 상자 위치를
workcell.yaml 의 예비값(origin_m·grid·wall_height_m)으로 쓴다 — 자동 검출이 실패한 장면에서 쓰인다.

    python tools/workcell_detect.py                                      # Piper Studio 공유 메모리에서 1장
    python tools/workcell_detect.py --snapshot data/field/1009_layout/snap_0000.npz
    python tools/workcell_detect.py --save configs/workcell.yaml         # 예비값으로 저장

컬러는 회색 카드 보정(color.card_json 또는 --card)을 적용한 뒤 판단한다.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aple.config import load_config  # noqa: E402
from aple.hal.calibration import base_to_pixel  # noqa: E402
from aple.hal.replay import load_snapshot  # noqa: E402
from aple.inspection.color import CardCalib, apply_gain  # noqa: E402
from aple.perception.geometry import rect_base  # noqa: E402
from aple.perception.workcell_detect import WorkcellDetection, auto_params, detect_workcell  # noqa: E402


def _rotated_corners(center, size, yaw) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy = size[0] / 2, size[1] / 2
    local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    return local @ np.array([[c, s], [-s, c]]) + np.asarray(center)


def _poly(img, corners_xy, z, snap, T, color, thick=2):
    pts = np.column_stack([corners_xy, np.full(len(corners_xy), z)])
    uv, zc = base_to_pixel(pts, snap.intrinsics, T)
    if np.all(zc > 0):
        cv2.polylines(img, [np.round(uv).astype(np.int32)], True, color, thick)
        return uv
    return None


def overlay(snap, T, det: WorkcellDetection, cfg) -> np.ndarray:
    img = snap.color.copy()
    colors = {"UPPER": (0, 0, 255), "MIDDLE": (0, 200, 255), "REJECT": (255, 0, 255)}
    for b in det.boxes:
        col = colors.get(b.name, (128, 128, 128)) if b.usable else (128, 128, 128)
        top = b.floor_z_m + max(b.wall_height_m, 0.0)
        uv = _poly(img, _rotated_corners(b.center_xy, b.outer_size_m, b.yaw_rad), top, snap, T, col, 2)
        if b.usable and b.origin_m is not None:
            pitch = float(cfg["workcell"]["boxes"].get(b.name, {}).get("cell_pitch_m", 0.09))
            rows, cols = b.grid
            x0, y0, z0 = b.origin_m
            for k in range(rows * cols):
                r, c = divmod(k, cols)
                _poly(img, rect_base((x0 + (c + 0.5) * pitch, y0 + (r + 0.5) * pitch), (pitch, pitch)), z0, snap, T, col, 1)
        if uv is not None:
            label = f"{b.name or '-'} {b.grid[0]}x{b.grid[1]}" if b.usable else "x"
            cv2.putText(img, label, tuple(np.round(uv.min(0) + [4, 16]).astype(int)), 0, 0.5, col, 2)
    if det.pick_center_m is not None:
        cx, cy, cz = det.pick_center_m
        uv = _poly(img, rect_base((cx, cy), det.pick_size_m), cz, snap, T, (0, 255, 0), 2)
        if uv is not None:
            cv2.putText(img, "PICK", tuple(np.round(uv.min(0) + [4, 16]).astype(int)), 0, 0.5, (0, 255, 0), 2)
    return img


def save_fallback(path: Path, det: WorkcellDetection, cfg: dict) -> list[str]:
    """workcell.yaml boxes.<NAME> 한 줄(흐름 매핑)을 검출값으로 바꾼다. 주석은 유지."""
    text = path.read_text(encoding="utf-8").split("\n")
    changed = []
    for name, b in det.named().items():
        box = {**cfg["workcell"]["boxes"][name], **b.box_cfg(float(cfg["workcell"]["boxes"][name]["cell_pitch_m"]))}
        flow = (f"{{grid: [{box['grid'][0]}, {box['grid'][1]}], cell_pitch_m: {box['cell_pitch_m']:g}, "
                f"wall_height_m: {box['wall_height_m']:g}, origin_m: [{', '.join(f'{v:g}' for v in box['origin_m'])}]}}")
        pat = re.compile(rf"^(\s+){name}:\s*\{{.*?\}}(\s*#.*)?$")
        for i, line in enumerate(text):
            m = pat.match(line)
            if m:
                text[i] = f"{m.group(1)}{name}:  {flow}  # [자동 검출 예비값 — tools/workcell_detect.py]"
                changed.append(text[i].strip())
                break
        else:
            raise KeyError(f"{path} 에 boxes.{name} 흐름 매핑 줄이 없다")
    path.write_text("\n".join(text), encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> WorkcellDetection:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", type=Path, help="스냅샷 .npz (없으면 Piper Studio 공유 메모리에서 1장)")
    ap.add_argument("--card", type=Path, help="card.json (기본: color.card_json)")
    ap.add_argument("--out", type=Path, default=ROOT / "logs" / "workcell_detect" / "overlay.png")
    ap.add_argument("--save", type=Path, help="찾은 상자를 이 workcell.yaml 의 예비값으로 쓴다")
    args = ap.parse_args(argv)

    cfg = load_config()
    T = cfg["workcell"].get("T_base_camera")
    if T is None:
        raise SystemExit("workcell.T_base_camera 가 없다 — 카메라–로봇 보정 먼저")
    T = np.asarray(T, float)
    if args.snapshot:
        snap = load_snapshot(args.snapshot)
    else:
        from aple.hal.live_snapshot import PiperLiveCamera
        snap = PiperLiveCamera.from_config(cfg)()
    card = args.card or (Path(cfg["color"]["card_json"]) if cfg["color"].get("card_json") else None)
    if card is not None:
        card = card if card.is_absolute() else ROOT / card
        snap = replace(snap, color=apply_gain(snap.color, CardCalib.from_dict(json.loads(card.read_text()))))
    else:
        print("[알림] 회색 카드 보정 없음 — 색 판단이 틀어질 수 있다 (color.card_json)")

    det = detect_workcell(snap, T, cfg)
    p = auto_params(cfg)
    print(f"이름 배정: {p['assign']} (영상 쪽 / base 축), 평면 구역 = {p['flat_zone']}")
    for b in det.boxes:
        state = b.name if b.usable else "제외"
        origin = "-" if b.origin_m is None else "(" + ", ".join(f"{v * 1000:.0f}" for v in b.origin_m) + ") mm"
        print(f"  {state:7s} 중심 ({b.center_xy[0] * 1000:.0f}, {b.center_xy[1] * 1000:.0f}) mm  "
              f"크기 {b.outer_size_m[0] * 1000:.0f}×{b.outer_size_m[1] * 1000:.0f} mm  yaw {math.degrees(b.yaw_rad):.1f}°  "
              f"바닥 z {b.floor_z_m * 1000:.0f} mm  벽 {b.wall_height_m * 1000:.0f} mm  칸 {b.grid[0]}×{b.grid[1]}  "
              f"원점 {origin}  {b.note}")
    if det.pick_center_m is not None:
        c, s = det.pick_center_m, det.pick_size_m
        print(f"  집는 영역 중심 ({c[0] * 1000:.0f}, {c[1] * 1000:.0f}, {c[2] * 1000:.0f}) mm  크기 {s[0] * 1000:.0f}×{s[1] * 1000:.0f} mm")
    for w in det.warnings:
        print(f"[경고] {w}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), overlay(snap, T, det, cfg))
    print(f"그림: {args.out}")
    if args.save:
        for line in save_fallback(args.save, det, cfg):
            print(f"  저장 {line}")
    return det


if __name__ == "__main__":
    main()
