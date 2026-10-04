"""직경 → 크기 구간 → 개구 폭·파지 높이·파지력 (설계서 grasp_profile_selector). 필수.

m     = ρ · (4/3)π r³
F_min = SF · m · (g + a_max) / (2μ),   F_min ≤ F < F_bruise 가 불가능하면 feasible=False
gripper_range_mm 밖이면 feasible=False
F → effort 는 역할 D의 FSR 보정표로 환산 (configs/size_bins.yaml gripper_calib)
"""

from __future__ import annotations

from aple.schema import AppleState, GraspProfile


def select(apple: AppleState, cfg: dict) -> GraspProfile:
    raise NotImplementedError
