"""pyrealsense2 흉내 — 장비 없이 촬영 도구의 흐름(설정 고정 → 저장)을 시험한다.

실제 SDK 동작(노출 단위, 수렴 시간 등)은 흉내 내지 않는다. 도구가 부르는 함수 이름과 순서만 맞춘다.
"""

from __future__ import annotations

import types

import numpy as np

from tests.synthetic import INTR, Apple, half_red, render

_COLOR, _DEPTH, _ = render([Apple((0.0, 0.0, 0.6), 0.04, half_red)], card_rect=(20, 20, 80, 60))


class _Enum:
    def __getattr__(self, name):
        return name


stream = _Enum()
format = _Enum()
camera_info = _Enum()
option = _Enum()


class _Range:
    def __init__(self, lo, hi):
        self.min, self.max = lo, hi


class _Sensor:
    def __init__(self):
        self.opts = {"exposure": 156.0, "gain": 64.0, "white_balance": 4600.0,
                     "enable_auto_exposure": 1.0, "enable_auto_white_balance": 1.0}

    def supports(self, opt):
        return opt in self.opts

    def get_option_range(self, opt):
        return _Range(0.0, 1.0) if opt.startswith("enable_") else _Range(1.0, 10000.0)   # 켜기/끄기 옵션은 0~1

    def get_option(self, opt):
        return self.opts[opt]

    def set_option(self, opt, v):
        self.opts[opt] = v

    def get_depth_scale(self):
        return 0.001


class _Device:
    color = _Sensor()

    def supports(self, info):
        return True

    def get_info(self, info):
        return {"usb_type_descriptor": "3.2", "serial_number": "000000000000", "name": "Intel RealSense D435",
                "firmware_version": "5.16.0.1"}[info]

    def first_color_sensor(self):
        return self.color

    def first_depth_sensor(self):
        return _Sensor()


class _Frame:
    def __init__(self, data):
        self.data = data

    def get_data(self):
        return self.data


class _Frames:
    def get_color_frame(self):
        return _Frame(_COLOR)

    def get_depth_frame(self):
        return _Frame((_DEPTH * 1000).astype(np.uint16))


class _Profile:
    def get_device(self):
        return _Device()

    def get_stream(self, s):
        intr = types.SimpleNamespace(fx=INTR[0], fy=INTR[1], ppx=INTR[2], ppy=INTR[3])
        return types.SimpleNamespace(as_video_stream_profile=lambda: types.SimpleNamespace(get_intrinsics=lambda: intr))


class pipeline:
    def start(self, cfg):
        return _Profile()

    def wait_for_frames(self, timeout_ms=5000):
        return _Frames()

    def stop(self):
        pass


class config:
    def enable_stream(self, *a):
        pass

    def enable_device(self, serial):
        pass


class align:
    def __init__(self, s):
        pass

    def process(self, frames):
        return frames


class context:
    def query_devices(self):
        return [_Device()]


class device:
    pass
