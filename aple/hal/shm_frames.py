"""Piper Studio 공유 메모리 프레임 읽기 (읽기 전용) — Piper Studio 를 켠 채로 카메라 영상을 받는다.

형식은 wego-piper-web `shm/piper_shm/segment.py`, `frames.py` (커밋 b311b22) 를 그대로 따른다.
Piper Studio 의 `piper_shm` 패키지를 import 하지 않아도 되도록 읽기 쪽만 옮겨 왔다
(우리 프로그램이 Piper Studio 와 다른 Python 환경에서 돌아도 동작하게).

세그먼트
  경로     /dev/shm/piper.cam.<카메라 id 의 ':' '/' 를 '_' 로>      예: rs:123456:color → piper.cam.rs_123456_color
  헤더     64 B, "<IIIIIIIQQQ" = magic("PIPC") ver w h ch dtype n_slots slot_bytes write_seq wall_ns
  슬롯     n_slots 개 원형 버퍼, 최신 프레임은 slot[write_seq % n_slots]
  동기화   seqlock — 읽기 전후 write_seq 차이가 n_slots−1 미만이면 유효

⚠ Piper Studio 가 형식(VERSION)을 바꾸면 여기서 즉시 실패한다 — 조용히 쓰레기 프레임을 읽지 않는다.
"""

from __future__ import annotations

import mmap
import struct
import time
from pathlib import Path

import numpy as np

SHM_DIR = Path("/dev/shm")
NAME_PREFIX = "piper.cam."
MAGIC = 0x50495043
VERSION = 1
HEADER_SIZE = 64
HEADER_FMT = "<IIIIIIIQQQ"
DTYPE_UINT8 = 1
_SEQ_OFF = struct.calcsize("<IIIIIIIQ")
_WALL_OFF = _SEQ_OFF + 8


class ShmError(RuntimeError):
    pass


def segment_path(cam_id: str, shm_dir: Path = SHM_DIR) -> Path:
    return Path(shm_dir) / (NAME_PREFIX + cam_id.replace(":", "_").replace("/", "_").strip("_"))


def find_realsense_serials(shm_dir: Path = SHM_DIR) -> list[str]:
    """컬러 세그먼트가 살아 있는 RealSense 시리얼 목록."""
    pre = NAME_PREFIX + "rs_"
    return sorted(p.name[len(pre):-len("_color")] for p in Path(shm_dir).glob(pre + "*_color"))


class ShmReader:
    def __init__(self, path: str | Path, max_retries: int = 8) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise ShmError(f"세그먼트가 없다: {self.path} — Piper Studio 카메라 데몬이 켜져 있는지 확인")
        self._f = open(self.path, "rb")
        self._buf = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        magic, ver, w, h, ch, dtype, n_slots, slot_bytes, _, _ = struct.unpack(
            HEADER_FMT, self._buf[:struct.calcsize(HEADER_FMT)])
        if magic != MAGIC:
            raise ShmError(f"세그먼트 형식이 다르다 (magic {magic:#x})")
        if ver != VERSION:
            raise ShmError(f"세그먼트 버전 {ver} — 이 리더는 {VERSION} 만 읽는다. Piper Studio 업데이트 내역 확인")
        if dtype != DTYPE_UINT8 or w * h * ch != slot_bytes:
            raise ShmError("세그먼트 크기 정보가 맞지 않는다")
        self.shape = (h, w, ch)
        self.n_slots = n_slots
        self.slot_bytes = slot_bytes
        self.max_retries = max_retries

    def _seq(self) -> int:
        return struct.unpack("<Q", self._buf[_SEQ_OFF:_SEQ_OFF + 8])[0]

    def read(self) -> tuple[np.ndarray, int, int] | None:
        """(프레임 복사본, seq, wall_ns). 아직 프레임이 없으면 None."""
        for _ in range(self.max_retries):
            s1 = self._seq()
            if s1 == 0:
                return None
            off = HEADER_SIZE + (s1 % self.n_slots) * self.slot_bytes
            frame = np.frombuffer(self._buf, np.uint8, count=self.slot_bytes, offset=off).reshape(self.shape).copy()
            wall = struct.unpack("<Q", self._buf[_WALL_OFF:_WALL_OFF + 8])[0]
            if self._seq() - s1 < self.n_slots - 1:
                return frame, s1, wall
        return None

    def close(self) -> None:
        self._buf.close()
        self._f.close()


def read_pair(color: ShmReader, depth: ShmReader, max_skew_ms: float = 50.0,
              timeout_s: float = 1.0) -> tuple[np.ndarray, np.ndarray, int] | None:
    """컬러·깊이를 거의 같은 시각의 것으로 한 쌍 읽는다 → (컬러, 인코딩 깊이, wall_ns).

    두 세그먼트는 따로 갱신되므로 시각 차이가 max_skew_ms 이내일 때만 받는다.
    판정은 팔이 멈춘 장면에서 하므로 이 정도 차이는 결과에 영향이 없다.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        c, d = color.read(), depth.read()
        if c is not None and d is not None and abs(c[2] - d[2]) <= max_skew_ms * 1e6:
            # Earliest exposure: freshness must hold for both color and depth.
            return c[0], d[0], min(c[2], d[2])
        time.sleep(0.005)
    return None
