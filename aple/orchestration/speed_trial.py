"""Operator advice after a fully verified live run; never changes speed settings."""
import math
from aple.schema import FailureCode, TaskAction


def speed_trial_advice(cfg, records, *, live_execution, cycles):
    if not live_execution or not records:
        return None
    expected = cfg['workcell']['expected_apple_count']
    if expected <= 0 or {r.cycle_index for r in records} != set(range(1, cycles+1)):
        return None
    actions = (TaskAction.PICK_PLACE, TaskAction.PICK_INSPECT_PLACE)
    for cycle in range(1, cycles+1):
        rows = [r for r in records if r.cycle_index == cycle]
        if len(rows) != expected or len({r.apple_id for r in rows}) != expected:
            return None
        for r in rows:
            if (r.apple_id <= 0 or not r.success or r.failure_code != FailureCode.NONE
                    or r.task_action not in actions or r.target_cell < 0
                    or not math.isfinite(r.settle_error_m)):
                return None
            times = (r.t_pick_s, r.t_place_s)
            if r.task_action == TaskAction.PICK_INSPECT_PLACE:
                times += (r.t_inspect_s,)
            if not all(math.isfinite(t) and t >= 0 for t in times):
                return None
    current = float(cfg['safety']['speed_scale_max'])
    steps = cfg['execution'].get('motion', {}).get('speed_trial_steps', [])
    if (not math.isfinite(current) or not 0 < current <= 1
            or not isinstance(steps, list) or not steps
            or any(isinstance(s, bool) or not isinstance(s, (int,float))
                   or not math.isfinite(s) or not 0 < s <= 1 for s in steps)
            or steps != sorted(set(steps))):
        return None
    next_scale = next((s for s in steps if s > current+1e-9), None)
    if next_scale is None:
        return None
    return (f'전체 실행·안착 확인 완료. 다음 속도 시험 후보: {current:g} → {next_scale:g}. '
            '사과의 미끄럼·손상을 현장에서 확인하고 같은 조건으로 반복 성공한 뒤, '
            f'safety.speed_scale_max={next_scale:g} 변경을 PR로 기록해 시험하세요. '
            '현재 배율은 속도와 가속도 한도에 함께 적용됩니다. '
            '빈손 배율·최종 하강·복구 설정은 별도이며, 배율은 자동 변경하지 않았습니다.')
