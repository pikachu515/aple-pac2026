"""이름 붙은 관절 자세(home·scan·hold·상자 위) 가르치기 — 텔레오퍼레이션으로 팔로워를 그 자세에 두고 관절각을 저장한다.

실기 실행(tools/run_cycle.py --live --execute)은 workcell.poses.home_joint_rad · scan_joint_rad 가 있어야 한다.
  home: 시작·종료·복구 자세 (팔이 탑뷰를 가리지 않고 테이블에서 충분히 높게)
  scan: 탑뷰 촬영 자세 (팔과 그림자가 트레이·상자를 가리지 않게)
  hold: ACT ① (집기·다면 보여주기) 가 사과를 들고 멈추는 자세
  hover_UPPER|hover_MIDDLE|hover_REJECT: 등급 상자 위 — 스크립트가 여기까지 옮기고 ACT ② 가 아래 빈 칸에 놓는다

    python tools/teach_pose.py home --iface can_follower --dry-run    # 지금 관절각만 보기
    python tools/teach_pose.py scan --iface can_follower              # configs/workcell.yaml 에 저장

관절각은 Piper Studio 에서 읽기만 한다 (팔을 움직이지 않는다). --iface 를 주면 이 실행에서만 읽기를 켜고 설정 파일은 그대로.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aple.config import load_config, set_config_value  # noqa: E402

POSES = ("home", "scan", "hold", "hover_UPPER", "hover_MIDDLE", "hover_REJECT")


def read_joints(cfg: dict, iface: str | None) -> tuple[float, ...]:
    from aple.hal.piper_studio import PiperStudio
    if iface:
        cfg["piper_web"] = {**cfg["piper_web"], "enabled": True, "iface": iface}
    return tuple(float(q) for q in PiperStudio(cfg).joint_state())


def check_limits(joints, limits) -> list[str]:
    """safety.joint_limits_rad 밖의 관절 (limits 가 없으면 빈 목록)."""
    if not limits:
        return []
    return [f"joint{i + 1} {math.degrees(q):.1f}° ∉ [{math.degrees(lo):.1f}, {math.degrees(hi):.1f}]"
            for i, (q, (lo, hi)) in enumerate(zip(joints, limits)) if not lo <= q <= hi]


def main(argv: list[str] | None = None, *, joint_reader=read_joints) -> tuple[float, ...]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pose", choices=POSES)
    ap.add_argument("--iface", help="관절각을 읽을 팔로워 (예: can_follower)")
    ap.add_argument("--workcell", type=Path, default=ROOT / "configs" / "workcell.yaml")
    ap.add_argument("--dry-run", action="store_true", help="저장하지 않고 보여 주기만")
    args = ap.parse_args(argv)

    cfg = load_config()
    joints = joint_reader(cfg, args.iface)
    if len(joints) != 6 or not all(math.isfinite(q) for q in joints):
        raise SystemExit(f"관절각이 이상하다: {joints}")
    print(f"{args.pose}: " + ", ".join(f"{math.degrees(q):.1f}°" for q in joints))
    bad = check_limits(joints, cfg["safety"].get("joint_limits_rad"))
    if bad:
        raise SystemExit("safety.joint_limits_rad 밖이다 — 자세를 바꾼다: " + "; ".join(bad))
    if args.dry_run:
        return joints
    line = set_config_value(str(args.workcell), f"workcell.poses.{args.pose}_joint_rad",
                            [round(q, 5) for q in joints])
    print(f"저장: {args.workcell}\n  {line.strip()}")
    return joints


if __name__ == "__main__":
    main()
