"""Read and verify Piper Studio v0.5.12 torque without moving or changing settings.

    python3 tools/check_gripper_torque.py

Use the configured follower and execution.gripper.fixed_effort. If authentication
is enabled, provide PIPER_WEB_PASSWORD in the environment (never in this file).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aple.config import load_config  # noqa: E402
from aple.hal.piper_studio import PiperStudio, PiperWebError  # noqa: E402


def main(*, cfg=None, make_studio=PiperStudio):
    config = copy.deepcopy(load_config() if cfg is None else cfg)
    # Enable only the read client for this command; no jog or write calls occur.
    config['piper_web']['enabled'] = True
    iface = config['piper_web'].get('iface')
    if not isinstance(iface, str) or not iface.strip() or 'PLACEHOLDER' in iface:
        raise SystemExit('piper_web.iface에 실제 등록된 follower 이름을 설정하세요')
    try:
        torque = make_studio(config).verify_gripper_torque()
    except (ValueError, PiperWebError) as exc:
        raise SystemExit(f'그리퍼 토크 확인 실패: {exc}') from exc
    print(f'{iface}: Studio {torque:g} N·m — execution.gripper.fixed_effort와 일치')
    print('장비 이동·토크 설정 변경 없음. 이 값은 설정 토크이며 측정 파지력(N)이 아닙니다.')
    return torque


if __name__ == '__main__':
    main()
