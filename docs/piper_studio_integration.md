# Piper Studio 연동 조사 — 카메라·깊이·회색 카드·조명 감시

M1·M2 판정 모듈이 대회에서 Piper Studio 영상을 받아 쓸 때 알아야 할 사실을 wego-piper-web 코드
(커밋 b311b22, 2026-09-29)에서 확인해 정리했다. 연동 코드는 `aple/hal/`.

## 1. 영상은 공유 메모리에서 읽는다 — 카메라를 따로 열지 않는다

| 항목 | 사실 | 근거 |
|---|---|---|
| 전달 방식 | 카메라 데몬(camerad·rsd)이 `/dev/shm` 세그먼트로 상시 발행, 여러 프로그램이 동시에 읽기 가능 | `shm/piper_shm/frames.py` (`Subscriber.read`, `read_new`) |
| D435 세그먼트 이름 | `rs:<시리얼>:color`, `rs:<시리얼>:depth`, (`rs:<시리얼>:infrared`) | `rs/piper_rs/hub.py` 프레임 루프 |
| 컬러 형식 | BGR uint8. **배경 마스크를 켜면 `far_mm` 너머 픽셀이 검게 칠해진 영상**이다 | `hub.py` `_mask_color` |
| 내부 파라미터 | rsd `intrinsics` RPC 로 fx·fy·cx·cy·해상도·왜곡 계수 제공 | `hub.py` `intrinsics()` |

녹화·정책이 읽는 바로 그 프레임을 판정 모듈도 읽게 된다.

## 2. 깊이는 그대로 쓸 수 없다 — 두 가지를 풀어야 한다

### 2-1. 컬러에 정렬되어 있지 않다

> "정렬한 깊이는 마스킹에만 쓴다. 발행하는 깊이는 원본 그대로다" — `hub.py` 주석

D435 깊이(왼쪽 IR 기준)와 컬러 센서는 약 15 mm 떨어져 있고 화각도 다르다. 깊이를 컬러에 그대로
겹치면 사과 영역이 어긋난다. 합성 장면 시험에서 정렬 없이 겹친 마스크는 실제 사과와 겹침 비율(IoU)이
0.5 미만이었다 (`tests/test_depth_align.py`).

### 2-2. 8비트로 인코딩되어 있다

`near_mm..far_mm` 를 0..254 로 선형 매핑(소수점 버림), 255 = 무효. 기본 150..1200 mm 이면 **한 단계 약 4 mm**.
인코딩 창은 카메라 API(`POST /api/cameras/{id}/depth-encoding`)로 바꿀 수 있다.

### 2-3. 깊이→컬러 외부 파라미터는 Piper Studio 가 주지 않는다

정렬에 필요한 깊이→컬러 회전·이동 값은 장치 공장 보정 상수다. **한 번 읽어 저장해 쓴다.**

```bash
systemctl --user stop piper-rsd
python tools/rs_check.py --save-calib configs/calib/d435.json
systemctl --user start piper-rsd
```

### 2-4. 해결 — `aple/hal/depth_align.py`

```python
from aple.hal.depth_align import D435Calib, DepthEncoding, snapshot_from_piper

calib = D435Calib.load("configs/calib/d435.json")
snap = snapshot_from_piper(color_bgr, depth_encoded, DepthEncoding(near_mm, far_mm), calib, scene_id)
# snap.depth_m 은 컬러 시점·컬러 해상도의 미터 깊이 → M1·M2 가 그대로 쓴다
```

합성 검증: 디코딩 오차 반 단계 이내, 정렬 깊이 오차 중앙값 2 mm 미만, 정렬 후 착색비율·직경이 정답과 일치.

## 3. 회색 카드 보정은 카메라 설정을 바꾸는 방식이다

| 항목 | 사실 | 근거 |
|---|---|---|
| 방식 | 노출(자동으로 찾은 뒤 고정) + **화이트밸런스 색온도(K)** + gain 을 장치에 설정. 소프트웨어 색 곱셈은 없다 | `feature/gray-card-calibration.md` §2 |
| 저장 | 카메라 프로파일로 저장하면 연결 시 자동 적용 | 같은 문서 §4 |
| 한계 | 색온도 하나로는 R/B 균형만 맞춘다. **초록–자주 방향 치우침은 남을 수 있다** | D435 RGB 의 WB 가 색온도 단일 값 |

**판정 모듈의 운용**: Piper Studio 보정을 먼저 하고, 시야에 둔 회색 카드로 우리 소프트웨어 보정
(`aple/inspection/color.py calibrate_card`)을 **한 번 더** 적용한다. 이미 맞춰진 영상이라 게인은 1 근처의
작은 값이 나오고, 남은 치우침만 잡는다. 카드가 시야에 있어야 하므로 **배경 마스크의 `far_mm` 은 테이블보다
멀게** 둔다 — 테이블이 검게 지워지면 카드도 지워진다.

## 4. 조명 감시

Piper Studio 조명 감시는 64×64 축소 영상의 밝기·log(R/G)·log(B/G)·3×3 격자 밝기를 빠른/느린 이동평균으로
비교해, 9칸 중 7칸 이상이 같은 방향으로 움직일 때만 경보한다 (`feature/lighting-watch.md`). 경보는 웹 화면
토스트로 나온다.

판정 모듈은 회색 카드 영역의 색 변화(ΔE)로 자체 판정한다 (`aple/inspection/color.py lighting_status`).
두 가지는 서로 보완한다 — Piper Studio 경보는 사람에게, 카드 ΔE 는 판정 보류(LOW_CONF) 결정에 쓴다.

## 5. 권장 설정 (상부 D435, 0.65 m)

| 설정 | 값 | 이유 |
|---|---|---|
| 깊이 인코딩 창 | **near 400 mm, far 900 mm** | 한 단계 약 2 mm (기본 4 mm). 사과 윗면 0.56 m · 테이블 0.65 m 가 여유 있게 들어간다 |
| 배경 마스크 `far_mm` | 테이블 거리 + 30 mm 이상 (예: 700 mm) | 테이블·회색 카드가 지워지지 않게 |
| 컬러·깊이 해상도 | 같게 (848×480) | 배경 마스크 동작 조건 (전략 S2-8) |
| 카메라 프로파일 | 회색 카드 보정 후 저장 | 재부팅 후에도 같은 색 |

깊이 인코딩 창을 바꾸면 그 창으로 녹화한 데이터셋과 이전 데이터셋의 깊이 의미가 달라진다. **녹화를 시작하기
전에 정하고 바꾸지 않는다.**

## 6. 남은 확인 (실기)

- [ ] `piper_shm.Subscriber` 로 `rs:<시리얼>:color` · `:depth` 를 같은 시각에 읽는 방법 (두 세그먼트의 시각 차)
- [ ] 우리 프로그램이 Piper Studio 와 같은 Python 환경에서 `piper_shm` 을 import 할 수 있는지 (설치 경로)
- [ ] 깊이 인코딩 창 현재값을 코드에서 읽는 방법 (`info()` 의 `depth_encoding`)
- [ ] 실제 D435 에서 `--save-calib` 값 확인 (이동 약 15 mm 근처인지)
