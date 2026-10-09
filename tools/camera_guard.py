"""카메라 프로파일 감시 — Piper Studio 가 노출을 기본값으로 되돌리면 활성 프로파일 값으로 다시 넣는다.

10/9 실기: Piper Studio(v0.5.12) 의 프로파일 적용(`profiles/apply`, 연결 시 자동 적용 포함)이
D435 컬러 노출을 넣지 못하고 장치 기본값(166 = 16.6 ms)으로 남긴다. 회색 카드로 맞춘 노출(10/9: 56)이
카메라를 다시 연결하거나 rsd 가 재시작될 때마다 풀려 영상이 과노출된다. 같은 값을
`POST /api/cameras/control` 로 하나씩 넣으면 들어간다.

고치는 조건 (버그의 흔적일 때만 손댄다)
  - 활성 프로파일이 그 카메라에 값을 정해 두었고
  - 지금 값이 **장치 기본값**이며, 프로파일 값은 기본값이 아니다
회색 카드 보정·사람이 손으로 바꾼 값(기본값이 아닌 값)은 건드리지 않는다.
자동 노출을 프로파일이 끄라고 했는데 켜져 있으면 그것도 끈다 (노출보다 먼저).

    python tools/camera_guard.py --once            # 한 번 점검·수정
    python tools/camera_guard.py                    # 계속 감시 (2초마다)
    python tools/camera_guard.py --dry-run --once   # 고칠 것만 보여 준다

상시 실행: tools/systemd/aple-camera-guard.service 를 ~/.config/systemd/user/ 에 두고
    systemctl --user daemon-reload && systemctl --user enable --now aple-camera-guard
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Callable
from urllib.parse import quote
from urllib.request import Request, urlopen

log = logging.getLogger("camera_guard")

GUARDED = ("exposure", "white_balance", "gain")          # 기본값으로 풀리면 되돌릴 값
AUTO_SWITCH = {"exposure": "enable_auto_exposure", "white_balance": "enable_auto_white_balance"}


def plan_fixes(cameras: list[dict], entries: list[dict],
               controls_of: Callable[[str], dict[str, dict]]) -> list[tuple[str, str, float]]:
    """→ [(cam_id, 컨트롤 이름, 넣을 값)]. 순수 함수 — controls_of(cam_id) 는 {이름: {"value", "default"}}.

    프로파일 항목과 카메라는 Piper Studio 와 같게 profile_key, 다음으로 last_dev(id) 로 짝짓는다.
    """
    fixes: list[tuple[str, str, float]] = []
    for cam in cameras:
        if not cam.get("connected"):
            continue
        entry = next((e for e in entries if e.get("key") == cam.get("profile_key")), None) or \
            next((e for e in entries if (e.get("match") or {}).get("last_dev") == cam.get("id")), None)
        wanted = (entry or {}).get("controls") or {}
        if not wanted:
            continue
        now = controls_of(cam["id"])
        for name in GUARDED:
            if name not in wanted or name not in now:
                continue
            want, cur, default = float(wanted[name]), now[name].get("value"), now[name].get("default")
            if cur is None or default is None or want == float(default) or float(cur) != float(default):
                continue
            switch = AUTO_SWITCH.get(name)
            if switch in wanted and switch in now and float(now[switch]["value"]) != float(wanted[switch]):
                fixes.append((cam["id"], switch, float(wanted[switch])))
            fixes.append((cam["id"], name, want))
    return fixes


class Studio:
    def __init__(self, base_url: str, timeout_s: float = 5.0) -> None:
        self.base, self.timeout = base_url.rstrip("/"), timeout_s

    def _req(self, method: str, path: str, body: dict | None = None):
        data = None if body is None else json.dumps(body).encode()
        req = Request(self.base + path, data=data, method=method, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read() or b"null")

    def cameras(self) -> list[dict]:
        return self._req("GET", "/api/cameras/current").get("cameras", [])

    def active_entries(self) -> list[dict]:
        name = self._req("GET", "/api/cameras/profiles/active").get("active") or ""
        if not name:
            return []
        preset = self._req("GET", f"/api/presets/camera/{quote(name, safe='')}")
        return list(((preset or {}).get("values") or {}).get("cameras") or [])

    def controls(self, cam_id: str) -> dict[str, dict]:
        d = self._req("GET", f"/api/cameras/{cam_id}/controls")
        items = d if isinstance(d, list) else d.get("controls", d)
        items = items.values() if isinstance(items, dict) else items
        return {c["name"]: c for c in items if isinstance(c, dict) and "name" in c}

    def set_control(self, cam_id: str, name: str, value: float) -> None:
        self._req("POST", "/api/cameras/control", {"id": cam_id, "name": name, "value": value})


def check_once(studio: Studio, dry_run: bool = False) -> list[tuple[str, str, float]]:
    fixes = plan_fixes(studio.cameras(), studio.active_entries(), studio.controls)
    for cam_id, name, value in fixes:
        if dry_run:
            log.info("[dry-run] %s %s → %g", cam_id, name, value)
            continue
        studio.set_control(cam_id, name, value)
        log.warning("%s %s 가 기본값으로 풀려 있어 프로파일 값 %g 로 되돌림", cam_id, name, value)
    return fixes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    studio = Studio(args.base_url)
    while True:
        try:
            fixes = check_once(studio, args.dry_run)
            if args.once:
                if not fixes:
                    log.info("고칠 것 없음")
                return 0
        except Exception as e:                          # Studio 재시작 중 등 — 감시는 계속한다
            log.warning("점검 실패: %s", e)
            if args.once:
                return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
