# 사과 몇 개로 데이터 모으기 (10/6 화)

실제 사과 수백 개는 불가능하다. 대신 **사과 한 개를 여러 번 다르게 쓰는 방법**으로 필요한 데이터를 만든다.
이 문서는 무엇을 몇 장 찍고, 어떤 도구로 어떻게 불리는지를 정리한다.

## 1. 결론부터

| 쓰임 | 담당 | 사과 5개로 되나 | 불리는 방법 |
|---|---|---|---|
| 착색 판정 기준값 | 변승호 | **된다** — 학습이 아니라 물리 기준값(색상각·오차폭)을 재는 것이라 개수보다 **색의 다양성**이 중요하다 | 사과마다 6자세 → **표본 30개** (`truth`) |
| YOLO 사과 검출 | 이수빈 | **된다** — 기본 모델(yolo11n, COCO)이 이미 '사과'를 안다. 우리 환경에 맞추는 미세조정만 하면 된다 | 배치 바꿔 수백 장 + **깊이 자동 라벨** + **붙여넣기 증강** |
| 크기 구간·파지력 | 강우담 | 일부 — 직경·무게 5개 실측 | AI Hub 크기·무게 데이터로 보완 (선택) |
| ACT 정책 | 서유겸 | 해당 없음 — 로봇이 있어야 한다 | 대회 당일 텔레옵 |

**사과를 더 살 수 있다면 10~15개를 권장한다.** 못난이·흠과 한 상자(1~2만 원)면 색이 제각각이라 오히려 좋다.
5개만 쓴다면 반드시 **빨간 것 · 반쯤 빨간 것 2개 · 노랗거나 초록빛인 것**을 섞는다.

## 2. 조사한 방법과 판단

| 방법 | 판단 | 이유 |
|---|---|---|
| **깊이 자동 라벨** (채택) | ◎ | D435 깊이로 테이블보다 솟은 덩어리 = 사과. 박스를 손으로 그리지 않는다. 붙은 사과 장면은 둥글기 검사로 자동 제외 (`tools/make_yolo_dataset.py`) |
| **붙여넣기 증강** (채택) | ◎ | 찍은 사과를 오려 빈 배경 사진에 무작위로 붙인다. 실제 사과 픽셀이라 색·질감이 진짜이고 라벨은 자동이다. 그림자는 부정확하므로 **학습에만** 섞고 검증은 실제 사진으로만 |
| **Piper Studio YOLO 루프** (채택) | ◎ | 위에서 만든 이미지·라벨을 REST 로 Piper Studio 데이터셋에 넣고(`--upload`) 화면에서 학습 → 가중치가 모델 목록에 자동 등록. 연동 경로 그대로 |
| Roboflow 공개 사과 데이터 | △ 선택 | 과수원·나무 배경이 대부분이라 우리 장면(테이블 위·수직 촬영)과 다르다. 기본 모델이 이미 사과를 알아 이득이 작다. **우리 데이터로 성능이 부족할 때만** 사전학습용으로 |
| AI Hub 농산물 품질 이미지 | △ 선택 | 승인이 필요하다. 착색 기준이 등급과 같은 방향인지 오프라인 확인용 (전략 S4-5-1) |
| 3D 렌더 합성 사과 | ✕ | 색·질감이 실제와 달라 착색 판정과 검출 모두에 해롭다 |
| 텔레옵 에피소드 프레임 가져오기 | 대회 당일 | 로봇이 있어야 생긴다. 생기면 같은 도구로 라벨 보강 |

## 3. 화요일 촬영 목록

D435 를 0.6~0.7 m 에서 수직으로 고정하고, **폴더 하나 = 촬영 세션 하나**로 나눠 찍는다.
도구는 `tools/capture_snapshot.py` (Piper Studio 끈 상태) 또는 `tools/judge_live.py` 의 **s** 저장 (Piper Studio 켠 상태).

| 폴더 | 내용 | 장수 | 쓰임 |
|---|---|---|---|
| `1006_desk` | 사과 5개 배치 바꿔 가며, 회색 카드 포함 | 30 | 착색 판정·검출 |
| `1006_repeat` | 한 배치를 **움직이지 않고** 연속 | 5~10 | 판정 재현성 (`repeat`) |
| `1006_yolo_a` | 1~5개, **화면 가장자리·구석**까지 위치 다양하게, 서로 3 cm 이상 띄워서 | 60 | 검출 학습 |
| `1006_yolo_b` | 조명 바꿔서 (스탠드 켜기·끄기, 창가 쪽), 바닥 바꿔서 (종이·트레이) | 60 | 검출 학습 |
| `1006_yolo_box` | 상자(골판지 등) 안·벽 근처에 놓은 사과 | 30 | 상자 속 사과 검출 (적재 점유용) |
| `1006_val` | **다른 시간·조명**에서 따로 찍은 장면 | 20 | 검증 전용 — 학습에 섞지 않는다 |
| `1006_empty` | 사과 없이 테이블·트레이·상자만 | 10 | 붙여넣기 증강 배경 |
| `truth/apple_01..05` | 사과 1개씩 6자세 (평소·뒤집기·눕혀 꼭지 축 90° ×4) | 30 | 탑뷰 오차·착색 정답 (`truth`) |

합계 약 250장, 2시간 안팎. 사과를 옮기는 동안 Piper Studio 캡처의 **N초 간격 연속 캡처**를 쓰면 더 빠르다
(단 그 경우 깊이가 저장되지 않으므로 자동 라벨은 judge_live/capture_snapshot 으로 찍은 장면만 가능).

**같이 기록할 것**: 사과 번호별 직경(캘리퍼스)·무게(저울), 육안 판정(상/중/애매) → `data/truth/labels.csv`

## 4. 찍은 뒤 처리

```bash
# 착색 판정 (docs/inspection_bench.md 8절)
python tools/color_session.py card    data/snapshots/1006_desk
python tools/color_session.py suggest data/snapshots/1006_desk --apply
python tools/color_session.py truth   data/truth --card data/snapshots/1006_desk/card.json --apply
python tools/color_session.py curve   data/truth/_analysis/truth_views.csv --apply     # 자세별 표본 30개
python tools/color_session.py repeat  data/snapshots/1006_repeat

# YOLO 데이터셋 — 자동 라벨 + 증강 300장, 검증은 1006_val 세션만
python tools/make_yolo_dataset.py data/snapshots/1006_desk data/snapshots/1006_yolo_* data/snapshots/1006_val \
    --val 1006_val --synth 300 --backgrounds data/snapshots/1006_empty --out data/yolo/apple_v1

# 같은 데이터를 Piper Studio 로 (화면에서 확인 → 학습 → 모델 목록에 등록)
python tools/make_yolo_dataset.py data/snapshots/1006_desk data/snapshots/1006_yolo_* data/snapshots/1006_val \
    --val 1006_val --synth 300 --backgrounds data/snapshots/1006_empty --upload http://localhost --dataset apple_v1
```

- 자동 라벨 결과는 **Piper Studio 라벨러 화면에서 한 번 훑어본다** (특히 상자 벽에 걸린 사과)
- 학습 기본 모델은 표준 `yolo11n` 부터. 증강 비율을 바꿔 가며(0 · 300 · 600장) 검증 세션 mAP 를 비교해 남긴다
- 찍은 폴더는 공유 드라이브 `snapshots/` 에, 데이터셋은 `yolo/` 에 올린다

## 5. 주의

- 깊이 자동 라벨은 **깊이가 있어야** 한다. USB 3 연결(`rs_check`)과 사과 간격(3 cm 이상)을 지킨다
- 증강 이미지는 그림자가 없다. 검증 세트에는 절대 넣지 않는다 (도구가 자동으로 학습 세트에만 넣는다)
- 같은 세션 연속 장면이 학습·검증에 나뉘면 점수가 부풀려진다. 검증 세션은 **다른 시간·조명**에서 따로 찍는다
