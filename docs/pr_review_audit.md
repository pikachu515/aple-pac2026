# 역할 B PR 요청 대조 (2026-10-07)

GitHub PR 1~25 중 존재하는 24개 PR의 본문·일반 댓글·리뷰·코드 줄 리뷰를 조회했다.
추가 페이지는 없었다. 아래는 현재 로컬 미커밋 변경과 대조한 결과다.
이 문서는 코드 대조 결과이며, GitHub 리뷰 스레드의 해결 표시를 대신하지 않는다.

| PR / 요청 | 코드 기준 결과 |
|---|---|
| [#11](https://github.com/pikachu515/aple-pac2026/pull/11): 기본 관절 한계의 임시값 제거 | 기존 반영: `joint_limits_rad: null` |
| #11: 그리퍼 timeout, 적재 상자 ID, NumPy 테스트 비교 | 기존 반영: HAL의 timeout 전달, `verify`의 ID 검사, `assert_allclose` |
| [#12](https://github.com/pikachu515/aple-pac2026/pull/12): TCP/FK 계약 | base_link·m 단위 TCP 계약 확인. 파지·촬영에 동일 FK 필요. 실제 TCP 오프셋·실측 검증은 남음 |
| [#15](https://github.com/pikachu515/aple-pac2026/pull/15)·[#16](https://github.com/pikachu515/aple-pac2026/pull/16): 즉시 검사·1회 재촬영 | 기존 미커밋 코드에 반영. 이번에 재촬영 직후 폭 감시도 추가 |
| #15: 첫 미끄러짐은 재파지, 두 번째는 리젝트 | 기존 내려놓기·리젝트 동작에 `execute_scene`의 횟수 관리·새 장면 재계획 호출부를 추가 |
| #15: 저속 복구·놓침 정지 | 기존 반영. 이번에 하강도 복구 속도를 초과하지 않게 하고 놓침을 DROP으로 기록 |
| [#19](https://github.com/pikachu515/aple-pac2026/pull/19): provider.begin·같은 FK | 기존 `inspection_begin` 훅 및 FK 불일치 경고 확인. 조립 시 아래처럼 주입 필요 |
| [#20](https://github.com/pikachu515/aple-pac2026/pull/20): snapshot/lighting/CycleManager 연결 지점 | HAL·CycleManager 주입 지점 확인. #23 helper와 #25 기반 SceneRunner를 연결해 FAULT 실행 차단 |
| [#25](https://github.com/pikachu515/aple-pac2026/pull/25): plan/execute·실패 기록 | head `9a4887b`의 SceneRunner를 로컬로 가져와 모드 B·backend·최종 등급/목표·phase 기반 실패 분류·빈 칸 없음 기록 보완 |

## 통합 브랜치의 공급자 연결 예시

#13·#19·#20·#21·#23이 병합된 최신 main(`630546e`)을 현재 작업 브랜치에 통합했다.
미커밋 변경을 임시 백업한 뒤 복원했고 #23 helper와 테스트의 중복·충돌을 정리했다.
아래 연결은 그 코드와 실측 설정·IK·구간 검증기가 준비된 뒤 사용한다.

```python
monitor = CardMonitor.from_config(cfg, live_camera)
ik, fk = build_piper_kinematics(cfg)  # tcp_offset_m 실측 필수, 같은 변환/geometry 사용
studio = PiperStudio(cfg, snapshot_provider=monitor.capture,
                     lighting_status_provider=monitor)
provider = InHandCaptureProvider.from_config(
    cfg, snapshot_fn=monitor.capture, tool_pose_fn=fk,
)
adapter = build_piper_motion_adapter(
    studio, cfg, ik, validator, tool_pose_provider=fk,
    in_hand_capture_provider=provider,
)
pipeline = PickPlacePipeline(cfg, adapter, inspection_begin=provider.begin)
runner = SceneRunner.from_config(
    cfg, pipeline, light_for=monitor.status_for,
    capture_for_retry=monitor.capture,
)
manager = CycleManager(
    cfg, capture=monitor.capture, plan_scene=runner.plan,
    execute_scene=runner.execute, make_records=runner.records,
)
```

재촬영은 새 FULL TOP scene_id를 반환하고 같은 apple_id를 추적해야 한다. 기존 탑뷰 등급을
새 위치에 결합하고 새 파지에서 apple_in_tcp를 다시 계산한다. 사과 재식별 실패·조명 FAULT면
재파지하지 않는다. 재촬영 후 이전 장면의 나머지 계획은 실행하지 않고 미실행 적재 예약을
해제한다. 다음 사이클에서 새로 계획한다. 실행 실패·예외 뒤에는 같은 SceneRunner에서 자동
재실행하지 않는다. 현장 원인 확인 후 실행기를 새로 구성해야 한다.

기록은 원래 사이클 scene_id와 실행된 최종 등급·상자·칸을 사용한다. 단계별 시간과 안착
오차는 측정값이 없어 NaN이다. #25 CLI의 실기 실행 잠금은 유지한다.

## 검증 및 남은 확인

- Python 구문 컴파일·이번에 수정한 실행/복구/통합 모듈의 pyflakes·`git diff --check` 확인.
- PR 커밋 전 필수 검증: `python3 -m pytest -q --tb=short` → **420 passed, 1 warning, 84.42s**.
  Python 3.10.12 / pytest 6.2.5에서 실행했다. 경고는 기존 pytest의 미지원 `pythonpath` 옵션이다.
  목표 Python 3.12 및 실기 검증을 대신하지 않는다.
- 실기·시뮬레이션, FK/TCP 오프셋, 그리퍼 폭 범위·놓침 기준·슬립 sigma 실측은 남음.
- #11 리뷰 스레드는 GitHub에서 미해결 표시지만 코드에는 이미 반영되어 있다.

## #25 역할 B 연결부 추가 확인

- 최신 main의 `detect(snap, cfg)`, `estimate(snap, dets, T, prev, cfg)`,
  `box_occupancy.estimate(snap, T, cfg)`, `plan_grasps(apples, cfg)`와 호출 형태가 일치한다.
- backend를 지정하지 않으면 `execution.default_backend`를 사용한다. 실제 ACT 구현은 별도다.
- 기록은 계획/실행의 scene_id와 사과 ID를 대조하며 중복 사과 실행 결과를 거부한다.
- 실패 매핑: 빈 칸 없음 P1, 접근·그리퍼 닫기 전 하강 F1, 파지·검사 F2,
  DROP F3, 이송·적재 F4. IDLE에서 실행 보류/게이트 차단은 물리 단계 실패로 집계하지 않는다.
- 모드 B 최종 배치 불가도 원래 계획이 아닌 최종 실행 항목 기준으로 P1을 기록한다.
- `.gitignore`에 `logs/`를 추가해 CSV 실행 기록을 커밋 대상에서 제외한다.

## TCP 오프셋 및 FK 보완

`execution.trajectory.build_piper_kinematics(cfg)`는 한 실측 `PiperTCPTransform`과 동일한
Piper geometry를 사용하는 IK/FK 쌍을 만든다. 설정은 플랜지 좌표계의 [x, y, z] m이며
TCP 축은 플랜지 축과 같다는 가정이다. 핑거 장착 후 역할 D가 실측한다. null/비정상 값은
어댑터 구성 전에 거부한다. 명시적 [0, 0, 0]은 영 오프셋 테스트/검증된 도구에만 사용한다.

- IK: `T_base_flange = T_base_tcp @ inv(T_flange_tcp)`를 Piper 수치 IK에 전달.
- FK: `link_transforms(q_rad[None, :], geometry)[0, tip_index] @ T_flange_tcp`를 Pose3D로 반환.
- `build_piper_motion_adapter`는 Piper IK 사용 시 설정값 및 FK 오프셋 일치를 확인한다.
- `joint_state()`가 이미 raw millidegree를 rad로 바꾸므로 FK에서 재변환하지 않는다.
- FK 객체는 파지 transport와 촬영 공급자에 동일하게 전달한다.
- 합성 해석 모델의 왕복/회전/영 오프셋·결측 잠금·잘못된 FK 변환 검증은 테스트로 확인한다.
  실제 Piper geometry와 수치 IK, TCP 실측의 검증을 대신하지 않는다.

Piper 공식 근거: [kinematics.py](https://github.com/WeGo-Robotics/wego-piper-web/blob/main/robot/piper_robot/kinematics.py).

## 시뮬레이션 실행 연결

최신 main의 #25/#26 실행 도구에 `--simulate`를 연결했다. 구간 안전 검사, named-pose 검사,
실제 Piper 수학 모델 오프셋 0/비영 왕복, 사과 3개의 판정→파지→저속 적재→기록을 검증한다.
실기 속도·데드맨·추가 툴/사과 및 자기 충돌 검증은 남아 있으며 `--live --execute`는 계속 거부한다.
모드 B·ACT·손상/마찰 동역학은 시뮬레이션 범위에 포함하지 않는다. [실행 방법](simulation.md).

## Piper Studio v0.5.12 토크 호환성

`PiperStudio.gripper_torque_nm`은 팔별 설정 토크(N·m)를 읽고,
`verify_gripper_torque`는 `execution.gripper.fixed_effort`의 SDK 단위(0.001 N·m)와 비교한다.
조그 시작·목표 전송·도달 대기에서 불일치/조회 실패를 거부하며 시작한 조그는 종료한다.
`tools/check_gripper_torque.py`는 이동/설정 변경 없이 확인만 한다. 자동 토크 조절·접촉 힘(N)
보정은 추가하지 않았다. 테스트는 localhost 가짜 0.5.12 서버 기준이며 실기 확인은 별도다.

## 수제 좌우 핑거 어댑터 설정

사용자가 확인한 사진의 점 사이 거리 48.144 / 48.464 / 71.76 / 79.76은 모두 mm로
기록했다. 장착 기하가 완성되면 같은 설정에서 TCP, HAL·명령 게이트의 접촉면 간격,
M1 파지 후보의 핑거 외형, 파지 높이, 구간 검사 추가 충돌 구를 구성한다.
기본 매뉴얼의 0–100 mm는 참조만 하며 실제 접촉면 간격으로 대입하지 않는다.
현재 미확정 장착 값은 null, installed는 false다. 활성화 후 누락 값은 거부한다.
[구체적인 설정 항목과 확인 방법](gripper_attachment.md).
