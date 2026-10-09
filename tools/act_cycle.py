"""스크립트 + ACT 실행 (10/10) — 탑뷰 판정 → ACT ① 집기·다면 → 등급 상자 위로 → ACT ② 놓기 → 접힌 자세.

    python tools/act_cycle.py --pick-model <ckpt> --place-model <ckpt> --items 1            # 실기 사과 1개
    python tools/act_cycle.py --pick-model <ckpt> --place-model <ckpt> --dry-run            # 촬영·판정·계획만

ckpt 는 Studio 컨테이너 안 경로 (예: /data/hf/hub/models--PIKACHUTT--act_pac2026_pick_inspect_v1/snapshots/<sha>).
판단 로직은 aple/orchestration/act_flow.py. 스크립트 이동은 계획된 저속 관절 궤적 (경로 충돌 검사 포함).
ACT 는 Studio 추론(/api/models/inference/start)으로 돌고, 팔로워 관절·그리퍼 폭을 보고 여기서 멈춘다.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aple.config import load_config  # noqa: E402
from aple.orchestration.act_flow import PickWatcher, PlaceWatcher, Sample, picked_grade  # noqa: E402

CAMERAS = {"top": "rs:243122071248:color", "wrist": "/dev/video10"}


_WS_LISTENER = """
import asyncio, json, sys, websockets
async def main():
    async with websockets.connect('ws://127.0.0.1:8000/ws',
            additional_headers={'Cookie': sys.argv[1], 'Origin': 'http://localhost'}) as ws:
        while True:
            m = json.loads(await ws.recv())
            if m.get('type') in ('log', 'state'):
                print(m['type'], m.get('data'), flush=True)
asyncio.run(main())
"""


class LogTap:
    """Studio 웹소켓에서 추론 로그를 받아 둔다 — 추론이 죽으면 이유를 보여 주려고 (Studio 는 로그를 남기지 않는다)."""

    def __init__(self, base_url="http://127.0.0.1"):
        self.lines = collections.deque(maxlen=300)
        self._proc = None
        self._base = base_url

    def start(self):
        req = urllib.request.Request(self._base + "/api/auth/login", method="POST",
                                     data=json.dumps({"password": os.environ.get("PIPER_WEB_PASSWORD", "")}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            cookie = "; ".join(c.split(";")[0] for c in r.headers.get_all("Set-Cookie") or [])
        self._proc = subprocess.Popen(["docker", "exec", "piper-web-backend", "/opt/venv/bin/python", "-c",
                                       _WS_LISTENER, cookie], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        threading.Thread(target=lambda: [self.lines.append(l.rstrip()) for l in self._proc.stdout], daemon=True).start()

    def tail(self, n=25):
        return "\n".join(list(self.lines)[-n:])

    def stop(self):
        if self._proc:
            self._proc.terminate()


class Act:
    """Studio 추론 시작·정지. 정지하면 팔로워는 그 자리에 선다 (robotd 데드맨)."""

    def __init__(self, studio, cfg, tap=None):
        self.studio, self.cfg, self.tap = studio, cfg, tap

    def running(self) -> bool:
        st = self.studio._request("GET", "/api/inference/status")
        return isinstance(st, dict) and st.get("state") == "running"

    def start(self, checkpoint: str) -> None:
        web = self.cfg["piper_web"]
        self.studio._request("POST", "/api/models/inference/start", {
            "checkpoint_path": checkpoint, "robot_type": "piper_follower", "robot_port": web["iface"],
            "camera_mapping": CAMERAS, "policy_type": "act"})
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            st = self.studio._request("GET", "/api/inference/status")
            if isinstance(st, dict) and st.get("state") == "running":
                return
            if isinstance(st, dict) and st.get("state") == "error":
                raise RuntimeError(f"ACT 추론 시작 실패: {st}\n{self.tap.tail() if self.tap else ''}")
            time.sleep(0.5)
        raise RuntimeError("ACT 추론이 60초 안에 시작되지 않았다")

    def stop(self) -> None:
        try:
            self.studio._request("POST", "/api/models/inference/stop", {})
        finally:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                st = self.studio._request("GET", "/api/inference/status")
                if not isinstance(st, dict) or st.get("state") in ("idle", "error", "stopped"):
                    return
                time.sleep(0.3)


def run_act(act, studio, checkpoint, watcher, timeout_s, width_of, period_s=0.1):
    """ACT 를 돌리며 watcher 가 끝이라고 할 때까지 (또는 시간 제한) 본다. → 끝났나."""
    act.start(checkpoint)
    t0 = time.monotonic()
    last_check = t0
    try:
        while time.monotonic() - t0 < timeout_s:
            if time.monotonic() - last_check > 1.0:
                last_check = time.monotonic()
                if not act.running():
                    raise RuntimeError("ACT 추론이 도중에 멈췄다 — Studio 로그:\n" + (act.tap.tail() if act.tap else "(없음)"))
            joints, g = studio._read_raw_state()
            if watcher.update(Sample(time.monotonic() - t0, joints, width_of(g))):
                return True
            time.sleep(period_s)
        return False
    finally:
        act.stop()
        settle(studio)


def settle(studio, still_rad=0.005, timeout_s=8.0):
    """ACT 를 멈춘 뒤 팔이 실제로 설 때까지 — 움직이는 중에 스크립트 이동을 걸면 속도 검사가 걸린다 (10/10 실기)."""
    prev, t0 = studio.joint_state(), time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        time.sleep(0.3)
        cur = studio.joint_state()
        if max(abs(a - b) for a, b in zip(cur, prev)) < still_rad:
            return
        prev = cur


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pick-model", required=True)
    ap.add_argument("--place-model", required=True)
    ap.add_argument("--items", type=int, default=1, help="이번 실행에서 옮길 사과 수")
    ap.add_argument("--pick-timeout", type=float, default=45.0)
    ap.add_argument("--place-timeout", type=float, default=20.0)
    ap.add_argument("--speed", type=float, default=0.3, help="스크립트 이동 속도 (safety.speed_scale_max 이하)")
    ap.add_argument("--dry-run", action="store_true", help="촬영·판정·이동 계획만 (팔을 움직이지 않는다)")
    args = ap.parse_args(argv)

    from aple.hal.piper_studio import PiperStudio
    from aple.hal.piper_motion import build_piper_motion_adapter
    from aple.hal.live_snapshot import PiperLiveCamera
    from aple.execution.segment_validator import build_piper_segment_validator
    from aple.execution.trajectory import build_piper_kinematics
    from aple.execution.gripper_geometry import apply_attachment_geometry
    from aple.execution.goal_builder import Pose3D
    from aple.execution.pick_place_pipeline import PickPlacePipeline
    from aple.orchestration.scene_runner import SceneRunner
    from tools.run_cycle import with_card_monitor

    cfg = load_config()
    cfg["piper_web"].update(enabled=True, robotd_safety_confirmed=True)
    cfg = apply_attachment_geometry(cfg)
    poses = cfg["workcell"]["poses"]
    hold, scan = tuple(poses["hold_joint_rad"]), tuple(poses["scan_joint_rad"])
    hover = {g: tuple(poses[f"hover_{g}_joint_rad"]) for g in ("UPPER", "MIDDLE", "REJECT")}
    closed, opened = cfg["piper_web"]["gripper_width_range_m"]
    width_of = lambda g: closed + g / 100.0 * (opened - closed)      # noqa: E731

    studio = PiperStudio(cfg)
    ik, fk = build_piper_kinematics(cfg)
    guard = build_piper_segment_validator(cfg, fk)
    adapter = build_piper_motion_adapter(studio, cfg, ik, guard, tool_pose_provider=fk)
    timeout = float(cfg["execution"]["timeouts"]["named_pose_s"])
    capture, light_for = with_card_monitor(cfg, PiperLiveCamera.from_config(cfg), None)
    runner = SceneRunner.from_config(cfg, PickPlacePipeline(cfg, None), light_for=light_for)
    tap = LogTap()
    tap.start()
    act = Act(studio, cfg, tap)

    def move(name, joints, payload_r=None):
        planner = adapter._planner
        if payload_r:
            guard.set_payload(Pose3D((0.0, 0.0, -payload_r), (0.0, 0.0, 0.0, 1.0)), payload_r)
        try:
            if args.dry_run:
                traj = planner.plan_joint_goal(joints, studio.joint_state())
                for point in traj.points_rad:                  # Studio 로 보낼 값까지 (관절 범위 변환)
                    studio._normalize_joint_target(point)
                print(f"  [계획] → {name}: 경로 통과")
                return True
            ok = adapter.move_named(name, joints, speed_scale=args.speed, timeout_s=timeout)
        finally:
            guard.clear_payload()
        print(f"  → {name}: {'도착' if ok else '실패'}")
        return ok

    done = 0
    while done < args.items:
        print(f"\n== 사과 {done + 1}/{args.items}")
        if not move("scan", scan):
            raise SystemExit("접힌 자세로 못 갔다 — 현장 확인")
        plan = runner.plan(capture())
        if not plan.apples:
            print("트레이에 사과가 없다 — 끝"); break
        for a in plan.apples:
            g = next((x.grade.name for x in plan.grades if x.apple_id == a.apple_id), "?")
            print(f"  사과 {a.apple_id}: ({a.center_m[0]:.3f}, {a.center_m[1]:.3f}) 직경 {a.diameter_m*1000:.0f} mm 등급 {g}")
        if args.dry_run:
            for name, q in (("hold", hold), *((f"hover_{g}", q) for g, q in hover.items())):
                move(name, q)
            break

        pick = PickWatcher(hold)        # 수제 핑거: 쥐어도 완전히 닫힘으로 읽혀 빈손 판정을 하지 않는다
        if not run_act(act, studio, args.pick_model, pick, args.pick_timeout, width_of):
            if pick.grasp_sample is None:
                print("  ACT ①: 시간 안에 못 집었다 → 접힌 자세로 돌아가 다시"); continue
            raise SystemExit("ACT ①: 집었지만 hold 자세에 오지 못했다 — 그 자리 정지, 사람 확인 (공중에서 열지 않는다)")
        tcp = fk(pick.grasp_sample.joints).position_m
        apple_id, grade = picked_grade(tcp[:2], plan.apples, plan.grades)
        print(f"  ACT ①: 집음 (핑거 끝 {tcp[0]:.3f}, {tcp[1]:.3f}) → 사과 {apple_id} 등급 {grade.name}, 쥔 폭 {pick.held_width_m*1000:.0f} mm")
        apple_r = next((a.diameter_m / 2 for a in plan.apples if a.apple_id == apple_id), 0.04)
        if not move(f"hover_{grade.name}", hover[grade.name], payload_r=apple_r):
            raise SystemExit("상자 위로 못 갔다 — 사과를 쥔 채 정지, 사람 확인")
        place = PlaceWatcher(pick.held_width_m)
        if not run_act(act, studio, args.place_model, place, args.place_timeout, width_of):
            raise SystemExit("ACT ②: 시간 안에 놓지 못했다 — 그 자리 정지, 사람 확인")
        print("  ACT ②: 놓음")
        done += 1
    move("scan", scan)
    print(f"\n완료: {done}개")


if __name__ == "__main__":
    main()
