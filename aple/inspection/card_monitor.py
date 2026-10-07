"""회색 카드로 스냅샷 색 보정 + 조명 상태 — capture 와 lighting_status_provider 를 한 번에. 역할 C.

Piper Studio 회색 카드 보정은 카메라 설정(노출·WB·게인)으로 색을 맞추지만 녹–자 틀어짐이 남을 수 있다.
그 위에 우리 카드 게인(`color.apply_gain`)을 얹고, 같은 프레임의 카드 색 변화(ΔE)로 조명 상태를 낸다.

  monitor = CardMonitor.from_config(cfg, live_camera)        # live_camera: 원본 SceneSnapshot 공급자
  CycleManager(capture=monitor.capture, ...)                  # 판정에는 보정된 스냅샷
  PiperStudio(lighting_status_provider=monitor, ...)          # 조명 상태 = 방금 찍은 그 프레임 기준
  inspect_scene(..., light=monitor.status_for(snapshot))      # 또는 스냅샷별로 직접

판정에 쓴 프레임과 조명을 확인한 프레임이 같아야 하므로, 상태는 capture 때 함께 계산해 둔다.
마지막 capture 가 max_age_s 보다 오래됐으면 새 프레임을 찍어 상태만 계산한다.
카드가 가려지면(팔·사과) 유효 픽셀이 부족해 FAULT(card_valid=False) — 판정은 LOW_CONF 로 보류된다.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

from aple.inspection.color import CardCalib, apply_gain, lighting_status
from aple.schema import LightingStatus, SceneSnapshot


class CardMonitor:
    def __init__(self, snapshot_fn: Callable[..., SceneSnapshot], calib: CardCalib, *,
                 delta_e_warn: float, delta_e_fault: float, max_age_s: float = 2.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if not 0 < delta_e_warn <= delta_e_fault:
            raise ValueError("need 0 < delta_e_warn <= delta_e_fault")
        self._snapshot_fn = snapshot_fn
        self.calib = calib
        self._warn, self._fault = float(delta_e_warn), float(delta_e_fault)
        self._max_age_s = float(max_age_s)
        self._clock = clock
        self._last: tuple[float, LightingStatus] | None = None
        self._by_scene: dict[int, LightingStatus] = {}

    @classmethod
    def from_config(cls, cfg: dict, snapshot_fn: Callable[..., SceneSnapshot],
                    card_path: str | Path | None = None, **kwargs) -> "CardMonitor":
        gc = cfg["color"]["gray_card"]
        path = card_path or cfg["color"].get("card_json")
        if not path:
            raise ValueError("color.card_json is not set — run tools/color_session.py card and point to card.json")
        calib = CardCalib.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        return cls(snapshot_fn, calib, delta_e_warn=float(gc["delta_e_warn"]),
                   delta_e_fault=float(gc["delta_e_fault"]), **kwargs)

    def capture(self, *args, **kwargs) -> SceneSnapshot:
        """원본 스냅샷 → 카드 게인 적용 스냅샷. 이 프레임의 조명 상태를 함께 기록한다."""
        raw = self._snapshot_fn(*args, **kwargs)
        status = self._measure(raw)
        self._by_scene[raw.scene_id] = status
        return replace(raw, color=apply_gain(raw.color, self.calib))

    def status_for(self, snapshot: SceneSnapshot) -> LightingStatus:
        """capture 로 받은 스냅샷의 조명 상태."""
        try:
            return self._by_scene[snapshot.scene_id]
        except KeyError:
            raise KeyError(f"scene {snapshot.scene_id} was not captured through this monitor") from None

    def __call__(self, *args, **kwargs) -> LightingStatus:
        """lighting_status_provider — 방금 찍은 프레임의 상태, 오래됐으면 새로 찍어서."""
        if self._last is not None and self._clock() - self._last[0] <= self._max_age_s:
            return self._last[1]
        return self._measure(self._snapshot_fn())

    def _measure(self, raw: SceneSnapshot) -> LightingStatus:
        status = lighting_status(raw.color, self.calib, self._warn, self._fault)
        self._last = (self._clock(), status)
        return status
