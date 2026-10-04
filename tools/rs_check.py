"""D435 점검 — 연결, USB 3 협상, 컬러·깊이 동시 스트림 fps, 깊이 유효율 (S2-7, S2-8).

케이블을 바꾸거나 다시 꽂을 때마다 실행한다.
Piper Studio 카메라 데몬이 D435를 점유 중이면 열리지 않으므로 멈춘 뒤 실행한다.

    python tools/rs_check.py
    python tools/rs_check.py --seconds 10 --width 640 --height 480
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pyrealsense2 as rs


def info(dev: rs.device, key: rs.camera_info) -> str:
    return dev.get_info(key) if dev.supports(key) else "-"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    devs = rs.context().query_devices()
    if len(devs) == 0:
        print("[FAIL] RealSense 장치가 없다 — 케이블·포트 확인, lsusb 로 인식 여부 확인")
        return 1

    ok = True
    dev = devs[0]
    usb = info(dev, rs.camera_info.usb_type_descriptor)
    serial = info(dev, rs.camera_info.serial_number)
    print(f"장치     : {info(dev, rs.camera_info.name)}")
    print(f"시리얼   : {serial}")
    print(f"펌웨어   : {info(dev, rs.camera_info.firmware_version)}")
    print(f"USB      : {usb}")
    if usb.startswith("2"):
        print("[FAIL] USB 2.0으로 협상됨 — USB 3.x 케이블·포트로 교체 (깊이 스트림 제한)")
        ok = False

    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    pipe = rs.pipeline()
    profile = pipe.start(cfg)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    try:
        for _ in range(30):                    # 자동 노출 안정화 대기
            pipe.wait_for_frames(timeout_ms=3000)

        n, valid, center = 0, [], []
        t0 = time.time()
        while time.time() - t0 < args.seconds:
            frames = pipe.wait_for_frames(timeout_ms=3000)
            depth = np.asanyarray(frames.get_depth_frame().get_data())
            valid.append(float((depth > 0).mean()))
            h, w = depth.shape
            c = depth[h // 2 - 10:h // 2 + 10, w // 2 - 10:w // 2 + 10]
            c = c[c > 0]
            if c.size:
                center.append(float(np.median(c)) * scale * 1000)
            n += 1
        fps = n / (time.time() - t0)
    finally:
        pipe.stop()

    if n == 0:
        print("[FAIL] 프레임을 받지 못했다 — 케이블 재연결, Piper Studio 카메라 데몬(piper-rsd) 중지 여부 확인")
        return 1

    print(f"스트림   : {args.width}x{args.height} 컬러+깊이, 실측 {fps:.1f} fps (목표 {args.fps})")
    print(f"깊이 유효: 평균 {np.mean(valid) * 100:.1f} % (프레임 전체)")
    if center:
        print(f"중앙 거리: {np.median(center):.0f} mm")
    if fps < args.fps * 0.9:
        print("[FAIL] fps 미달 — USB 대역폭·컨트롤러 배치 확인 (S2-7)")
        ok = False

    print("[OK] 점검 통과" if ok else "[FAIL] 점검 실패 항목 있음")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
