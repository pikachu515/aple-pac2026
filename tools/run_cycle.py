"""전체 사이클 실행 — 카메라 → 인식 → 판정 → 계획 → (실행) → 기록. M4 통합.

    python tools/run_cycle.py --replay data/snapshots/1006_desk     # 저장한 스냅샷으로 드라이런 (로봇 안 움직임)
    python tools/run_cycle.py --live --cycles 3                      # Piper 카메라로 판정·계획까지 (로봇 안 움직임)
    python tools/run_cycle.py --live --execute --cycles 1            # 실제 로봇 실행 (빠진 부품이 있으면 거부)

현장에서 바로 고치기 (시뮬레이션 결과는 참고만 — 실제 장면으로 확인한다)
    python tools/run_cycle.py --live --save data/field/1009_a         # 찍은 장면을 그대로 저장 (원본 색)
    python tools/run_cycle.py --replay data/field/1009_a --set inspection.slip_check.sigma_axial_m=0.004
                                                                     # 같은 실제 장면을 값만 바꿔 다시 돌려 본다 (파일은 그대로)
    값이 정해지면 tools/field_values.py set 으로 설정 파일에 써 넣는다

장면마다 사과별 판정·작업·목표 칸을 표로 보여 주고, orchestration.logging.csv_dir 에 cycles.csv 를 남긴다.
조명이 FAULT 면 실행하지 않고 "회색 카드 재보정"을 안내한다 (전략 S3-8).

필요한 것
  M1 인식·캘리브레이션 (#13): aple.perception.*, workcell.T_base_camera
  카드 보정 (#20): color.card_json 이 있으면 CardMonitor 로 색 보정 + 조명 상태. 없으면 보정 없이 진행
  실행(--execute): 순기구학 공급자·TCP 오프셋(#24), 구간 안전 검사기(SegmentValidator)
"""

from __future__ import annotations

import argparse
import sys
from functools import reduce
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from aple.config import load_config  # noqa: E402
from aple.execution.pick_place_pipeline import PickPlacePipeline  # noqa: E402
from aple.hal.replay import load_snapshot  # noqa: E402
from aple.orchestration.cycle_manager import CycleManager  # noqa: E402
from aple.orchestration.scene_runner import SceneRunner  # noqa: E402
from aple.schema import CycleRecord, SceneSnapshot  # noqa: E402


def replay_source(folder: Path) -> Callable[[], SceneSnapshot]:
    files = sorted(folder.glob("*.npz"))
    if not files:
        raise SystemExit(f"{folder} 에 스냅샷(.npz)이 없다")
    state = {"i": 0}

    def capture() -> SceneSnapshot:
        i = state["i"]
        state["i"] += 1
        return load_snapshot(files[i % len(files)], scene_id=i + 1)

    capture.count = len(files)  # type: ignore[attr-defined]
    return capture


def saving(capture: Callable[[], SceneSnapshot], folder: Path) -> Callable[[], SceneSnapshot]:
    """찍은 원본 스냅샷을 folder 에 .npz 로 남긴다 (replay·color_session 으로 다시 분석 가능)."""
    folder.mkdir(parents=True, exist_ok=True)

    def capture_and_save() -> SceneSnapshot:
        snap = capture()
        np.savez_compressed(folder / f"snap_{snap.scene_id:04d}.npz", color=snap.color, depth_m=snap.depth_m,
                            intrinsics=np.array(snap.intrinsics), stamp_s=snap.stamp_s)
        return snap

    return capture_and_save


def apply_overrides(cfg: dict, items: list[str]) -> list[str]:
    """'a.b.c=값' → 이번 실행의 cfg 만 바꾼다 (파일은 그대로). 없는 키는 오타로 보고 거부한다."""
    applied = []
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--set 은 키=값 형태다: {item}")
        *parents, leaf = key.strip().split(".")
        try:
            node = reduce(lambda d, k: d[k], parents, cfg)
            old = node[leaf]
        except (KeyError, TypeError):
            raise SystemExit(f"--set: 설정에 {key} 가 없다 (오타 확인)") from None
        node[leaf] = yaml.safe_load(raw)
        applied.append(f"{key}: {old} -> {node[leaf]}")
    return applied


def with_card_monitor(cfg: dict, capture: Callable[[], SceneSnapshot], card_path: Path | None):
    """card.json 이 있으면 CardMonitor 로 감싼다 → (capture, light_for). 없으면 보정 없이."""
    path = card_path or cfg["color"].get("card_json")
    if not path or not Path(path).exists():
        print("  회색 카드 보정값(card.json) 없음 - 색 보정·조명 감시 없이 진행")
        return capture, None
    try:
        from aple.inspection.card_monitor import CardMonitor
    except ImportError:
        print("  CardMonitor(#20) 가 아직 없음 - 색 보정·조명 감시 없이 진행")
        return capture, None
    monitor = CardMonitor.from_config(cfg, capture, card_path=path)
    print(f"  회색 카드 보정: {path}")
    return monitor.capture, monitor.status_for


def missing_for_execution(cfg: dict) -> list[str]:
    """실제 로봇 실행 전에 채워져 있어야 하는 것들. 코드 부품이 들어오면 이 목록과 실행 어댑터 연결을 함께 고친다."""
    missing = []
    if cfg["workcell"].get("T_base_camera") is None:
        missing.append("workcell.T_base_camera - 카메라-로봇 캘리브레이션 (역할 A, #13 tools/touch_calib.py)")
    if cfg["workcell"].get("tcp_offset_m") is None:
        missing.append("workcell.tcp_offset_m - 핑거 어댑터 장착 후 TCP 오프셋 실측 (역할 D, 이슈 #24)")
    missing.append("실행 어댑터 연결 - 순기구학 tool_pose_provider, IK 의 TCP->플랜지 변환, "
                   "구간 안전 검사기 SegmentValidator (역할 B, 이슈 #24). 들어오면 run_cycle 에 연결한다")
    return missing


def print_records(records: list[CycleRecord]) -> None:
    print(f"  {'scene':>5} {'사과':>4} {'직경mm':>6} {'착색비율 [구간]':>20} {'등급':>6} {'사유':>9} {'작업':>18} {'목표':>10} {'결과':>4}")
    for r in records:
        target = f"{r.target_box.name}#{r.target_cell}" if r.target_cell >= 0 else "-"
        print(f"  {r.scene_id:>5} {r.apple_id:>4} {r.diameter_m * 1000:>6.1f} "
              f"{r.ratio:>6.2f} [{r.ratio_lo:.2f}, {r.ratio_hi:.2f}] {r.grade.name:>6} {r.reject_reason.name:>9} "
              f"{r.task_action.name:>18} {target:>10} {'성공' if r.success else '-':>4}")


def main(argv: list[str] | None = None, *, make_runner: Callable[[dict, PickPlacePipeline], SceneRunner] | None = None,
         make_live_camera: Callable[[dict], Callable[[], SceneSnapshot]] | None = None) -> list[CycleRecord]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--replay", type=Path, help="스냅샷(.npz) 폴더로 드라이런")
    src.add_argument("--live", action="store_true", help="Piper Studio 카메라 (공유 메모리)")
    ap.add_argument("--execute", action="store_true", help="실제 로봇 실행 (--live 에서만)")
    ap.add_argument("--cycles", type=int, help="사이클 수 (기본: replay 는 파일 수, live 는 1)")
    ap.add_argument("--card", type=Path, help="card.json (기본: color.card_json 또는 replay 폴더의 card.json)")
    ap.add_argument("--save", type=Path, help="찍은 원본 장면을 이 폴더에 저장 (나중에 --replay 로 재검증)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="이번 실행에만 설정값을 바꾼다 (반복 가능, 파일은 그대로)")
    args = ap.parse_args(argv)

    cfg = load_config()
    for line in apply_overrides(cfg, args.set):
        print(f"  임시 변경 {line}")
    if args.execute and not args.live:
        raise SystemExit("--execute 는 --live 와 함께만 쓴다")
    if args.execute:
        missing = missing_for_execution(cfg)
        if missing:
            print("실제 로봇 실행을 거부한다 - 아래가 아직 없다:")
            for m in missing:
                print(f"  - {m}")
            raise SystemExit(2)

    if args.replay:
        capture = replay_source(args.replay)
        cycles = args.cycles or capture.count  # type: ignore[attr-defined]
        card = args.card or (args.replay / "card.json" if (args.replay / "card.json").exists() else None)
    else:
        if make_live_camera is None:
            from aple.hal.live_snapshot import PiperLiveCamera
            make_live_camera = PiperLiveCamera.from_config
        capture = make_live_camera(cfg)
        cycles = args.cycles or 1
        card = args.card

    print(f"{'실행' if args.execute else '드라이런'} - {cycles} 사이클")
    if args.save:
        capture = saving(capture, args.save)
        print(f"  장면 저장: {args.save}")
    capture, light_for = with_card_monitor(cfg, capture, card)
    pipeline = PickPlacePipeline(cfg)            # 실행 어댑터는 missing_for_execution 이 비면 여기서 연결한다
    runner = (make_runner or (lambda c, p: SceneRunner.from_config(c, p, light_for=light_for)))(cfg, pipeline)

    manager = CycleManager(cfg, capture=capture, plan_scene=runner.plan, execute_scene=runner.execute,
                           make_records=runner.records, dry_run_capture=capture, dry_run=not args.execute)
    records = manager.run(cycles)
    print_records(records)
    blocked = [r for r in records if r.failure_code.name == "ABORTED" and r.target_cell < 0]
    if blocked:
        print("※ 실행하지 않은 장면이 있다 - 조명 FAULT 면 회색 카드 재보정 후 다시 (전략 S3-8)")
    return records


if __name__ == "__main__":
    main()
