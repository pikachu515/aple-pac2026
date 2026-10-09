"""회색 카드로 스냅샷 색 보정 + 조명 상태 — capture 와 lighting_status_provider 를 한 번에. 역할 C.

Piper Studio 회색 카드 보정은 카메라 설정(노출·WB·게인)으로 색을 맞추지만 녹–자 틀어짐이 남을 수 있다.
그 위에 우리 카드 게인(`color.apply_gain`)을 얹고, 같은 프레임의 카드 색 변화(ΔE)로 조명 상태를 낸다.

  monitor = CardMonitor.from_config(cfg, live_camera)        # live_camera: 원본 SceneSnapshot 공급자
  CycleManager(capture=monitor.capture, ...)                  # 판정에는 보정된 스냅샷
  PiperStudio(lighting_status_provider=monitor, ...)          # 조명 상태 = 방금 찍은 그 프레임 기준
  inspect_scene(..., light=monitor.status_for(snapshot))      # 또는 스냅샷별로 직접

판정에 쓴 프레임과 조명을 확인한 프레임이 같아야 하므로, 상태는 capture 때 함께 계산해 둔다.
레퍼런스 색 패치(patch.json)가 있으면 조명이 틀어졌을 때만 3×3 패치 보정을 시도하고(ref_patch, 전략 S3-5),
보정 후 패치가 기준과 맞으면 상태를 CORRECTING 으로 낮춰 판정을 이어 간다. 안 맞으면 FAULT 그대로.
마지막 capture 가 max_age_s 보다 오래됐으면 새 프레임을 찍어 상태만 계산한다.
카드 면 깊이(card.json card_depth_m)가 있으면 카드보다 카메라 쪽 픽셀(회색 그리퍼·팔)은 카드로 치지 않는다.
카드가 가려지면(min_visible 미만) 조명 변화가 아니므로, occluded_reuse_s 안에 마지막으로 카드가 보였던
프레임의 상태를 그대로 쓴다 (모드 B 처럼 팔이 화면에 있을 때 — 조명은 몇 초 사이에 바뀌지 않는다).
그보다 오래됐거나 깊이 기준이 없으면 FAULT(card_valid=False) — 판정은 LOW_CONF 로 보류된다.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

from aple.inspection.color import CARD_DEPTH_TOL_M, CARD_MIN_VISIBLE, CardCalib, apply_gain, lighting_status
from aple.inspection.ref_patch import PatchCalib, apply_matrix, patch_correction, patch_delta_e
from aple.schema import LightingStatus, LightState, SceneSnapshot


class CardMonitor:
    def __init__(self, snapshot_fn: Callable[..., SceneSnapshot], calib: CardCalib, *,
                 delta_e_warn: float, delta_e_fault: float, patches: PatchCalib | None = None,
                 max_age_s: float = 2.0, depth_tol_m: float = CARD_DEPTH_TOL_M,
                 min_visible: float = CARD_MIN_VISIBLE, occluded_reuse_s: float = 0.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if not 0 < delta_e_warn <= delta_e_fault:
            raise ValueError("need 0 < delta_e_warn <= delta_e_fault")
        self._snapshot_fn = snapshot_fn
        self.calib = calib
        self.patches = patches
        self._warn, self._fault = float(delta_e_warn), float(delta_e_fault)
        self._max_age_s = float(max_age_s)
        self._depth_tol_m, self._min_visible = float(depth_tol_m), float(min_visible)
        self._occluded_reuse_s = float(occluded_reuse_s)
        self._last_visible: tuple[float, LightingStatus] | None = None
        self._clock = clock
        self._last: tuple[float, LightingStatus] | None = None
        self._by_scene: dict[int, LightingStatus] = {}

    @classmethod
    def from_config(cls, cfg: dict, snapshot_fn: Callable[..., SceneSnapshot],
                    card_path: str | Path | None = None, patch_path: str | Path | None = None,
                    **kwargs) -> "CardMonitor":
        gc = cfg["color"]["gray_card"]
        path = card_path or cfg["color"].get("card_json")
        if not path:
            raise ValueError("color.card_json is not set — run tools/color_session.py card and point to card.json")
        calib = CardCalib.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        ppath = patch_path or cfg["color"].get("patch_json")
        patches = PatchCalib.from_dict(json.loads(Path(ppath).read_text(encoding="utf-8"))) if ppath else None
        occ = {"depth_tol_m": float(gc["depth_tol_m"]), "min_visible": float(gc["min_visible"]),
               "occluded_reuse_s": float(gc["occluded_reuse_s"])}
        return cls(snapshot_fn, calib, delta_e_warn=float(gc["delta_e_warn"]),
                   delta_e_fault=float(gc["delta_e_fault"]), patches=patches, **{**occ, **kwargs})

    def capture(self, *args, **kwargs) -> SceneSnapshot:
        """원본 스냅샷 → 카드 게인 적용 스냅샷. 이 프레임의 조명 상태를 함께 기록한다."""
        raw = self._snapshot_fn(*args, **kwargs)
        color, status = self._measure(raw)
        self._by_scene[raw.scene_id] = status
        return replace(raw, color=apply_gain(color, self.calib))

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
        return self._measure(self._snapshot_fn())[1]

    def _measure(self, raw: SceneSnapshot) -> tuple[np.ndarray, LightingStatus]:
        """→ (판정에 쓸 원본 색: 필요하면 패치 보정, 조명 상태)."""
        color = raw.color
        now = self._clock()
        status = lighting_status(color, self.calib, self._warn, self._fault, raw.depth_m,
                                 tol_m=self._depth_tol_m, min_visible=self._min_visible)
        if not status.card_valid and self._last_visible is not None \
                and now - self._last_visible[0] <= self._occluded_reuse_s:
            status = self._last_visible[1]                  # 카드가 가려졌을 뿐 — 직전에 본 조명 상태
            self._last = (now, status)
            return color, status
        if status.light_state != LightState.OK and self.patches is not None:
            M = patch_correction(color, self.patches)
            if M is not None:
                corrected = apply_matrix(color, M)
                if patch_delta_e(corrected, self.patches) <= self._warn:
                    color = corrected
                    status = LightingStatus(LightState.CORRECTING, status.delta_e, status.card_valid)
        if status.card_valid:
            self._last_visible = (now, status)
        self._last = (now, status)
        return color, status
