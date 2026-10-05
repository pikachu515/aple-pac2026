"""D435 스냅샷 저장 — 로봇 없이 M1·M2를 개발하기 위한 데이터 (컬러 + 컬러 정렬 깊이 + 내부 파라미터).

자동 노출·자동 화이트밸런스를 끄고 고정한 상태로 찍는다 (전략 S2-4·S2-5). 색 보정은 나중에
tools/color_session.py card 로 회색 카드에서 계산하므로, 여기서는 "모든 장이 같은 설정"인 것만 중요하다.

Piper Studio가 D435를 열고 있으면 실행되지 않는다. Piper Studio 카메라(RealSense) 데몬을 멈춘 뒤 실행한다.

미리보기 창 키
  s       저장                      c   회색 카드 영역 선택 (밝기 표시용)
  [ / ]   노출 내림 / 올림           n   다음 사과 폴더로 (apple_01 → apple_02, 카메라 설정 유지)
  q       종료

    python tools/capture_snapshot.py --out data/snapshots/1006_desk --note "형광등, 사과 5개"
    python tools/capture_snapshot.py --out data/truth/apple_01         # 사과 1개, 자세를 바꿔 가며 6장
    python tools/capture_snapshot.py --out ... --no-preview            # 화면 없이 Enter 로 저장
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

rs = None    # pyrealsense2 — main() 에서 불러온다 (도구 함수 시험 시 D435 드라이버 없이 import 가능)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aple.inspection.color import TARGET_GRAY_SRGB, card_brightness  # noqa: E402


def next_folder(path: Path) -> Path:
    """끝 숫자를 1 올린 형제 폴더 (자릿수 유지). 숫자가 없으면 _2 를 붙인다."""
    m = re.search(r"(\d+)$", path.name)
    if not m:
        return path.with_name(path.name + "_2")
    num = str(int(m.group(1)) + 1).zfill(len(m.group(1)))
    return path.with_name(path.name[:m.start()] + num)


def set_opt(sensor, opt, value) -> None:
    if sensor.supports(opt):
        lo = sensor.get_option_range(opt)
        sensor.set_option(opt, float(min(max(value, lo.min), lo.max)))


def get_opt(sensor, opt):
    return sensor.get_option(opt) if sensor.supports(opt) else None


def lock_color(sensor, exposure: float | None, wb: float) -> dict:
    """자동 노출·자동 WB 끄기. exposure 가 None 이면 자동 노출이 정한 값을 읽어 그대로 고정한다."""
    if exposure is None:
        exposure = get_opt(sensor, rs.option.exposure)
    set_opt(sensor, rs.option.enable_auto_exposure, 0)
    if exposure is not None:
        set_opt(sensor, rs.option.exposure, exposure)
    set_opt(sensor, rs.option.enable_auto_white_balance, 0)
    set_opt(sensor, rs.option.white_balance, wb)
    return {"exposure": get_opt(sensor, rs.option.exposure), "gain": get_opt(sensor, rs.option.gain),
            "white_balance": get_opt(sensor, rs.option.white_balance)}


def main() -> None:
    global rs
    import pyrealsense2
    rs = pyrealsense2

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--exposure", type=float, default=None, help="고정 노출값 (기본: 자동 노출 수렴값을 읽어 고정)")
    ap.add_argument("--wb", type=float, default=4600, help="고정 화이트밸런스 (색은 회색 카드로 나중에 보정)")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    pipe = rs.pipeline()
    profile = pipe.start(cfg)
    dev = profile.get_device()
    usb = dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else "?"
    if usb.startswith("2"):
        print(f"[경고] USB {usb} 로 연결됨 - USB 3 케이블·포트로 바꾸지 않으면 깊이가 불안정하다")
    color_sensor = dev.first_color_sensor()
    scale = dev.first_depth_sensor().get_depth_scale()
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    align = rs.align(rs.stream.color)

    for _ in range(45):                          # 자동 노출 수렴 대기 후 고정
        pipe.wait_for_frames()
    locked = lock_color(color_sensor, args.exposure, args.wb)
    print("카메라 설정 고정:", locked)

    meta = {"serial": dev.get_info(rs.camera_info.serial_number), "usb": usb,
            "width": args.width, "height": args.height,
            "intrinsics": [intr.fx, intr.fy, intr.ppx, intr.ppy], "camera": locked, "note": args.note}

    def save(frames, out: Path, n: int) -> None:
        color = np.asanyarray(frames.get_color_frame().get_data()).copy()
        depth_m = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * scale
        path = out / f"snap_{n:04d}.npz"
        np.savez_compressed(path, color=color, depth_m=depth_m,
                            intrinsics=np.array(meta["intrinsics"]), stamp_s=time.time())
        meta["camera"] = {"exposure": get_opt(color_sensor, rs.option.exposure),
                          "gain": get_opt(color_sensor, rs.option.gain),
                          "white_balance": get_opt(color_sensor, rs.option.white_balance)}
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  저장: {path}  (깊이 유효 {(depth_m > 0).mean() * 100:.0f}%)")

    n = len(list(out.glob("snap_*.npz")))
    try:
        if args.no_preview:
            while input(f"[{n}] Enter = 저장, q = 종료 > ").strip().lower() != "q":
                save(align.process(pipe.wait_for_frames()), out, n)
                n += 1
            return

        import cv2
        card = None
        while True:
            frames = align.process(pipe.wait_for_frames())
            color = np.asanyarray(frames.get_color_frame().get_data())
            view = color.copy()
            exp = get_opt(color_sensor, rs.option.exposure)
            text = f"{out.name} #{n}  exposure {exp:.0f}" if exp is not None else f"{out.name} #{n}"
            if card is not None:
                x, y, w, h = card
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text += f"  card {card_brightness(color, card):.0f} (target {TARGET_GRAY_SRGB:.0f})"
            cv2.putText(view, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(view, "s save  n next apple  c card  [ ] exposure  q quit", (10, view.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow("capture", view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("s"):
                save(frames, out, n)
                n += 1
            elif k == ord("n"):
                out = next_folder(out)
                out.mkdir(parents=True, exist_ok=True)
                n = len(list(out.glob("snap_*.npz")))
                print(f"다음 폴더: {out}")
            elif k == ord("c"):
                r = cv2.selectROI("capture", color, showCrosshair=False)
                card = tuple(int(v) for v in r) if r[2] > 0 and r[3] > 0 else None
            elif k in (ord("["), ord("]")) and exp is not None:
                set_opt(color_sensor, rs.option.exposure, exp * (0.85 if k == ord("[") else 1.18))
        cv2.destroyAllWindows()
    finally:
        pipe.stop()


if __name__ == "__main__":
    main()
