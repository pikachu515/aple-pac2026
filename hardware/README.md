# 하드웨어 — 역할 D

| 폴더 | 내용 |
|---|---|
| `cad/` | 핑거 오프셋 어댑터, 연성 패드 금형, 카메라 거치대 (STEP 원본 + STL) |
| `fsr_calib/` | FSR 측정 회로도, 아두이노 스케치, 토크–힘 보정 분석 스크립트 |
| `udev/` | D435i IMU 접근 권한 규칙 (`tools/rs_calibrate.py --height-mm` 에 필요, 역할 A) |
| `rs_calib/` | D435i 깊이 보정표 백업 — 10/8 Tare 저장 전 원본 (`tools/rs_calibrate.py restore 파일 --write`, 역할 A) |

기준 형상: AgileX `piper_ros` 의 그리퍼 모델 `link7.STL`, `link8.STL`

출력물을 바꿀 때마다 파일명에 오프셋 값과 버전을 넣는다. 예: `finger_adapter_o15_v2.step`
