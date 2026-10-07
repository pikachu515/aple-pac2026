# 장치 없는 Piper 동작 시뮬레이션

```bash
# Piper Studio 저장소의 robot/ 안에 piper_robot/kinematics.py와 data/arm_geometry.npz가 있어야 한다.
PYTHONPATH=/path/to/wego-piper-web/robot python3 tools/run_cycle.py \
  --simulate --trace logs/simulation/trace.json
```

이 환경에서 검증한 수학 모듈은 `/tmp/aple-piper-math`에 있다. 따라서 지금은
`PYTHONPATH=/tmp/aple-piper-math python3 tools/run_cycle.py --simulate --trace logs/simulation/trace.json`
으로 실행할 수 있다. /tmp는 임시 경로이므로 이후에는 설치된 Piper Studio의 robot 경로를 쓴다.
외부 코드는 저장소에 복사하거나 커밋하지 않는다.

`configs/simulation/piper.yaml`을 기존 설정에 **메모리에서만** 덮어쓴다. 가상 TCP·카메라·상자·사과·관절 범위는
실측값이 아니며 `configs/workcell.yaml`의 null은 유지한다. `load_config()`는 이 하위 폴더를 읽지 않으므로
다른 도구에 가상값이 섞이지 않는다. `--simulation-config 파일`로 별도 가상 셀을 지정할 수 있다.
`--set 키=값`은 덮어쓴 뒤 적용한다. 실기 `--live --execute`는 검증된 실기 연결 전까지 거부한다.
`--simulate --execute`는 허용하지 않으며 가상 전송은 HTTP·카메라·CAN을 열지 않는다.

## 실행 범위와 출력

합성 RGB-D → 실제 M1 검출·구 피팅·상자 점유·파지 후보 → 실제 M2 판정 → M5 계획 →
Piper 수치 IK·구간 검사·가상 전송 → Phase B 저속 놓기 → CSV 기록을 실행한다.
실행 전에는 같은 TCP 정의로 FK→IK→FK를 검사하며, seed를 바꾸어 수치 IK가 실제로 계산하게 한다.
세 자세의 위치·회전 잔차가 허용 범위를 넘으면 중단한다.

기본 장면은 직경 64 mm 사과 3개이고 중·경계 리젝트·상 판정이 각 상자에 연결된다.
가상 파지/놓기와 최종 사과 위치는 trace JSON, 판정·목표 칸·성공/실패는 `logs/simulation/cycles.csv`에 남긴다.
실행 실패가 있으면 CLI가 실패 종료한다. 가상 이동 시간은 관절 속도 모델로 산출한 값이며 실제 처리 시간과 다르다.
여러 사이클은 새 사과를 채우지 않고 이동한 뒤 장면을 다시 관측한다.

## 구간 안전 검사

`PiperSegmentValidator`는 시작점·끝점·중간 관절값을 `sample_step_rad` 간격 이하로 검사한다.
관절 제한, TCP 작업 범위, Piper 이동 링크의 복셀 덮개 구와 바닥·설정 AABB 충돌을 검사한다.
고정 base_link는 바닥 검사에서 제외한다. home/scan 이동도 같은 검사와 관절 분할을 거친다.
설정·기구학 출력이 유효하지 않으면 실행을 허용하지 않는다.

이는 **샘플링 검사**이므로 연속 구간 충돌 없음을 증명하지 않는다. 추가 어댑터는
장착 설정의 충돌 구를 입력하고 활성화하면 검사에 포함된다([장착 설정](gripper_attachment.md)).
기본 가상 셀에서는 어댑터가 비활성화되어 있으며, 자기 충돌·손에 든 사과의 충돌은
별도 모델링이 필요하다. 기본 가상 장면에는 별도 장애물 AABB가 없고
상자 벽은 계획 높이에만 반영되므로, 장애물 시험은 `safety.segment_validation.obstacle_aabbs_m`을 지정한다.

그리퍼는 이상적인 구 파지·놓기와 바닥으로의 즉시 안착을 가정한다. 마찰·힘·변형·낙하·손상·동역학과
Piper Web 조그의 속도·데드맨은 검증하지 않는다. 모드 B 촬영과 ACT는 지원하지 않으며 설정 단계에서 거부한다.
가상 TCP 자세 `[0,180,0]` RPY는 이 셀에서 도달 가능함을 확인한 설정이다.
현장 기본 `[180,0,0]`과 다르므로 실제 핑거 축 정의를 확인한 뒤 실측 셀에 적용해야 한다.

## 테스트

```bash
PYTHONPATH=/path/to/wego-piper-web/robot python3 -m pytest -q tests/test_simulation.py tests/test_segment_validator.py
```

Piper 수학 모듈이 없으면 해당 모델 테스트 3개는 skip된다. 장애물/작업 범위/바닥/관절 제한과
named-pose 검사 테스트는 외부 모듈 없이 실행된다.
실제 모델 검증에는 [Piper 공식 기구학](https://github.com/WeGo-Robotics/wego-piper-web/blob/587d1aa7dd00450b51c036839c80b65e557da360/robot/piper_robot/kinematics.py)과
같은 커밋의 arm_geometry.npz를 사용했다. 오프셋 0/0.16 m 왕복과 사과 3개 전체 실행을 검증했다.
