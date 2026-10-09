"""전체 사이클 실행 — 카메라 → 인식 → 판정 → 계획 → (실행) → 기록. M4 통합.

    python tools/run_cycle.py --replay data/snapshots/1006_desk     # 저장한 스냅샷으로 드라이런 (로봇 안 움직임)
    python tools/run_cycle.py --live --cycles 3                      # Piper 카메라로 판정·계획까지 (로봇 안 움직임)
    PYTHONPATH=/path/to/wego-piper-web/robot python tools/run_cycle.py --simulate --trace logs/simulation/trace.json
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
  가상 실행(--simulate): Piper 수학 모듈 + configs/simulation/piper.yaml
  실기 실행(--execute): 실측 설정과 실기 속도·충돌 검증 및 전송 연결 필요
"""

from __future__ import annotations

import argparse
import json
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
    from aple.hal.live_runtime import missing_for_live_execution
    return missing_for_live_execution(cfg)


def print_records(records: list[CycleRecord]) -> None:
    print(f"  {'scene':>5} {'사과':>4} {'직경mm':>6} {'착색비율 [구간]':>20} {'등급':>6} {'사유':>9} {'작업':>18} {'목표':>10} {'결과':>4}")
    for r in records:
        target = f"{r.target_box.name}#{r.target_cell}" if r.target_cell >= 0 else "-"
        print(f"  {r.scene_id:>5} {r.apple_id:>4} {r.diameter_m * 1000:>6.1f} "
              f"{r.ratio:>6.2f} [{r.ratio_lo:.2f}, {r.ratio_hi:.2f}] {r.grade.name:>6} {r.reject_reason.name:>9} "
              f"{r.task_action.name:>18} {target:>10} {'성공' if r.success else r.failure_code.name:>4}")


def main(argv: list[str] | None = None, *, make_runner: Callable[[dict, PickPlacePipeline], SceneRunner] | None = None,
         make_live_camera: Callable[[dict], Callable[[], SceneSnapshot]] | None = None) -> list[CycleRecord]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--replay", type=Path, help="스냅샷(.npz) 폴더로 드라이런")
    src.add_argument("--simulate", action="store_true", help="장치 없는 실제 Piper IK/FK 시뮬레이션")
    src.add_argument("--live", action="store_true", help="Piper Studio 카메라 (공유 메모리)")
    ap.add_argument("--simulation-config", type=Path, default=ROOT / "configs/simulation/piper.yaml")
    ap.add_argument("--trace", type=Path, help="시뮬레이션 이동·파지·놓기 및 왕복 검증 JSON")
    ap.add_argument("--execute", action="store_true", help="실제 로봇 실행 (--live 에서만)")
    ap.add_argument("--cycles", type=int, help="사이클 수 (기본: replay 는 파일 수, live 는 1)")
    ap.add_argument("--card", type=Path, help="card.json (기본: color.card_json 또는 replay 폴더의 card.json)")
    ap.add_argument("--save", type=Path, help="찍은 원본 장면을 이 폴더에 저장 (나중에 --replay 로 재검증)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="이번 실행에만 설정값을 바꾼다 (반복 가능, 파일은 그대로)")
    args = ap.parse_args(argv)
    if args.cycles is not None and args.cycles <= 0:
        ap.error("--cycles must be positive")

    cfg = load_config()
    if args.card is not None:
        cfg['color']['card_json'] = str(args.card)
    world = adapter = None
    checks = []
    if args.simulate:
        from aple.hal.simulation import simulation_config
        cfg = simulation_config(cfg, args.simulation_config)
    elif args.trace:
        ap.error("--trace requires --simulate")
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

    if args.simulate:
        from aple.hal.simulation import build_simulation
        try:
            world, adapter, checks = build_simulation(cfg)
        except ImportError as exc:
            raise SystemExit(f"Piper 수학 모듈 로딩 실패: {exc}. Piper Studio의 robot 폴더를 PYTHONPATH에 추가하세요") from exc
        except (ValueError, RuntimeError) as exc:
            raise SystemExit(f"시뮬레이션 구성 실패: {exc}") from exc
        capture, cycles, card = world.capture, args.cycles or 1, None
        print("Piper 수치 IK/FK 왕복 검증 완료 (TCP 오프셋 포함)")
    elif args.replay:
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

    print(f"{'시뮬레이션' if args.simulate else '실행' if args.execute else '드라이런'} - {cycles} 사이클")
    if args.save:
        capture = saving(capture, args.save)
        print(f"  장면 저장: {args.save}")
    capture, light_for = with_card_monitor(cfg, capture, card)
    if args.execute:
        from aple.hal.live_runtime import build_live_runtime
        runtime = build_live_runtime(cfg, capture, light_for)
        capture, pipeline, runner = runtime.capture, runtime.pipeline, runtime.runner
    else:
        pipeline = PickPlacePipeline(cfg, adapter)
        runner = (make_runner or (lambda c, p: SceneRunner.from_config(c, p, light_for=light_for)))(cfg, pipeline)

    manager = CycleManager(cfg, capture=capture, plan_scene=runner.plan, execute_scene=runner.execute,
                           make_records=runner.records, dry_run_capture=capture, dry_run=not (args.execute or args.simulate))
    records = manager.run(cycles)
    print_records(records)
    blocked = [r for r in records if r.failure_code.name == "ABORTED" and r.target_cell < 0]
    if blocked:
        print("※ 실행하지 않은 장면이 있다 - 조명 FAULT 면 회색 카드 재보정 후 다시 (전략 S3-8)")
    if world is not None:
        print(f"  가상 이동 {sum(e['action'] == 'move' for e in world.events)}회, "
              f"파지 {sum(e['action'] == 'grasp' for e in world.events)}회, "
              f"놓기 {sum(e['action'] == 'release' for e in world.events)}회, "
              f"가상 이동 시간 {world.elapsed_s:.2f}s")
        if args.trace:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            args.trace.write_text(json.dumps(dict(roundtrip=checks, events=world.events,
                                                apples=world.apples), indent=2), encoding="utf-8")
        if any(not r.success for r in records if r.task_action.name in ('PICK_PLACE', 'PICK_INSPECT_PLACE')):
            raise SystemExit("시뮬레이션 실행 실패: 위 결과와 trace를 확인하세요")
    return records


if __name__ == "__main__":
    main()
