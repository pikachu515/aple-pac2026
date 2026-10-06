# 색상 판정 실측 절차 (역할 C · D435 단독)

로봇 없이 D435와 사과만으로 M2(색상 품질 판별)의 기준값을 정하는 절차다.
같은 날 YOLO 검출용 촬영까지 함께 하려면 [data_collection.md](data_collection.md) 의 촬영 목록을 따른다.
결과로 `configs/varieties.yaml` 의 **[임시]** 값 세 가지를 실측값으로 바꾼다.

| 정할 값 | 도구 | 의미 |
|---|---|---|
| `hue_range_deg` 위쪽 경계 | `color_session.py suggest` | 착색 / 비착색을 가르는 색상각 |
| `top_view_error` | `color_session.py truth` | 탑뷰만 보고 판정할 때 전표면과의 오차 → 판정 구간 폭 |
| `delta_up` | `color_session.py curve` | '상' 판정 여유 — 리젝트율 vs 남은 오분류율 곡선에서 고른다 |
| (확인) `min_chroma`, 하이라이트 | `analyze` 오버레이 | 그림자·반사가 착색으로 잘못 세지지 않는지 |

## 0. 준비물

- [ ] 노트북 (Ubuntu, 저장소 최신, `pytest` 통과 확인 — 1절)
- [ ] D435 + USB 3 케이블 (예비 1개)
- [ ] 카메라를 0.6~0.7 m 높이에서 **수직으로 내려다보게** 고정할 것 (거치대, 삼각대, 선반 + 테이프)
- [ ] 회색 카드
- [ ] 사과 15~20개 — 많이 빨간 것, **반쯤 빨간 것(가장 중요)**, 노랗거나 초록빛인 것을 섞어서. 홍로·부사 섞기
- [ ] 번호 스티커 (꼭지 오목한 곳에 붙여 색을 가리지 않게), 펜·종이
- [ ] 병뚜껑 등 작은 받침 (사과를 옆으로 눕혀 세울 때)
- [ ] 무광 바닥 (회색·흰 종이 또는 무광 테이블) — 광택 있는 바닥은 깊이가 비고 반사가 생긴다
- [ ] (있으면) 스탠드 조명 1개 — 조명 변화 실험용, 캘리퍼스·저울 — 역할 D 데이터

## 1. 전날 집에서 (노트북 준비)

```bash
cd aple-pac2026
git switch main && git pull
source .venv/bin/activate
pip install -r requirements.txt
pytest                                   # 전부 통과해야 한다
python -c "import pyrealsense2; print('ok')"
```

## 2. 카메라 연결 확인 (5분)

```bash
systemctl --user stop piper-rsd          # Piper Studio가 D435를 잡고 있으면 우리 도구가 못 연다
python tools/rs_check.py                 # USB 3.2, 30 fps 근처, [OK] 가 나와야 한다
```

- `USB 2.1` 이 나오면 케이블·포트를 바꾼다 (깊이가 불안정해진다)
- **처음 한 번만**: 깊이→컬러 보정값을 저장한다 (Piper Studio 영상을 쓸 때 필요 — `docs/piper_studio_integration.md`)

```bash
python tools/rs_check.py --save-calib configs/calib/d435.json
```

  저장된 파일은 PR 로 올린다 (장치 고유 상수, 비밀 정보 아님)
- 실험이 끝나면 `systemctl --user start piper-rsd` 로 되돌린다

## 3. 배치

```
        D435 (0.6~0.7 m, 수직 아래)
            │
 ┌──────────┼──────────────────┐
 │ [회색카드]                    │   카드: 바닥에 평평하게, 그림자 없이, 화면 구석
 │                              │
 │    🍎      🍎      🍎        │   사과: 서로 3 cm 이상 띄우기 (붙으면 한 덩어리로 잡힌다)
 │                              │
 │    🍎      🍎                │   번호는 위 줄 왼→오, 아래 줄 왼→오 순서로 매겨진다
 └──────────────────────────────┘
```

## 4. 노출 맞추기 (5분)

```bash
python tools/capture_snapshot.py --out data/snapshots/1006_desk --note "실험실 형광등"
```

1. 미리보기 창에서 **c** → 회색 카드를 드래그 → Enter
2. 화면 위 `card ___ (target 118)` 이 **100~140** 이 되도록 **[ / ]** 로 노출 조절
3. 이후 이 세션에서는 노출·조명을 바꾸지 않는다

## 5. 세션 A — 탑뷰 판정 사진 (20분)

같은 창에서 계속 찍는다 (**s** = 저장).

1. 사과 5개를 놓고 **s** 2번 (같은 배치 2장)
2. 사과를 바꿔 가며 반복 → 모든 사과가 한 번 이상 찍히게
3. **종이에 "사진 번호 — 자리별 사과 번호"를 적는다** (예: `snap_0000, 0001: 위 줄 3·7·12 / 아래 줄 5·9`)
4. 한 배치는 사과를 움직이지 않고 **5장 이상** 찍어 `data/snapshots/1006_repeat` 에 따로 둔다 → 판정 재현성(⑦)

## 6. 세션 B — 전표면 정답 사진 (사과당 2분)

탑뷰 판정이 실제 전체 표면과 얼마나 다른지 재기 위한 사진이다. **사과 1개씩** 폴더를 따로 만든다.

```bash
python tools/capture_snapshot.py --out data/truth/apple_01
```

| 장 | 자세 |
|---|---|
| 1 | **평소대로 놓은 자세** (꼭지 위) — 탑뷰 판정에 쓰는 장 |
| 2 | 뒤집기 (꼭지 아래) |
| 3~6 | 옆으로 눕혀 받침에 세우고, **꼭지–꽃받침 축을 중심으로 90°씩 굴려** 4장 |

> ⚠ 3~6장에서 눕힌 사과를 **테이블 위에서 제자리로 빙글 돌리기만 하면 같은 면이 계속 위를 향한다.** 꼭지 축을 굴려서 위를 보는 면이 매번 바뀌어야 한다.

같은 노출을 유지하려면 미리보기 창을 닫지 말고, 사과마다 `--out` 만 바꿔 다시 실행할 때 `--exposure` 에 4절에서 맞춘 값을 넣는다 (창 위에 표시된 값).

## 7. 육안 판정표 (사진 분석 전에!)

결과를 보기 전에 적어야 판단이 섞이지 않는다. `data/truth/labels.csv`:

```
apple,label
apple_01,상
apple_02,중
apple_03,애매
```

가능하면 두 사람이 따로 적고, 의견이 다르면 `애매` 로 둔다.

## 8. 분석

```bash
# ① 색 보정값 — 회색 카드 영역 드래그
python tools/color_session.py card    data/snapshots/1006_desk

# ② 착색비율·등급 — _analysis/ 의 오버레이 이미지를 꼭 눈으로 본다
python tools/color_session.py analyze data/snapshots/1006_desk

# ③ 착색 경계 제안 → 설정 파일에 바로 반영(--apply) → ② 다시
python tools/color_session.py suggest data/snapshots/1006_desk --apply

# ④ 전표면 정답 → top_view_error 반영
python tools/color_session.py truth   data/truth --card data/snapshots/1006_desk/card.json --apply

# ⑤ 육안 판정과 비교
python tools/color_session.py compare data/truth/_analysis/truth.csv --labels data/truth/labels.csv

# ⑥ 리젝트 곡선 → delta_up 반영(제안값 + 여유 0.03) + 발표 그래프(_analysis/reject_curve.png)
python tools/color_session.py curve   data/truth/_analysis/truth_views.csv --labels data/truth/labels.csv --apply   # 자세별 표본

# ⑦ 판정 재현성 — 5절에서 같은 배치로 찍은 장면만 따로 모은 폴더 (사과를 움직이지 않고 5장 이상)
python tools/color_session.py repeat  data/snapshots/1006_repeat
```

**⑥ curve 결과 읽는 법**
- 제안 `delta_up` 은 "'중→상' 오류 0 중 리젝트가 가장 적은 점"이다. 표본이 적으면 경계에 딱 붙은 값이 나오므로 `--apply` 는 **제안값에 여유 0.03 을 더해** 쓴다 (`--margin` 으로 조절)
- `--apply` 로 바뀐 줄은 주석의 `[임시]` 가 `[실측 날짜]` 로 바뀐다. `git diff configs/varieties.yaml` 로 확인 후 PR
- `[중요] 탑뷰만으로는 '중→상' 오류를 ... 만들 수 없다` 가 나오면, 윗면만 빨간 사과가 있다는 뜻이다. 그 사과 이름이 함께 출력된다 → 다면 검사(모드 B)가 필요한 근거이므로 사진과 함께 기록해 둔다

**오버레이 보는 법** — 빨강 = 착색으로 판정, 파랑 = 비착색, 칠해지지 않은 가장자리 = 비스듬해서 판정 제외

| 보이는 것 | 의미 | 조치 |
|---|---|---|
| 사과가 아예 안 잡힘 / 두 개가 한 덩어리 | 분할 실패 | 사과 간격 넓히기, 깊이 유효율(`rs_check`) 확인 |
| 그림자 진 부분이 빨강 | 어두운 부분 오판 | `min_chroma` 올리기 |
| 반짝이는 부분이 파랑 | 하이라이트가 판정에 들어감 | `highlight.min_L` 내리기 |
| 눈으로 빨간데 파랑, 주황 경계가 애매 | 경계 색상각 | ③ suggest 값 확인, 육안표와 맞춰 미세 조정 |

## 9. (시간 남으면) 조명 변화 실험

1. 스탠드 조명을 켜거나 위치를 바꾼 뒤 같은 사과를 `data/snapshots/1006_lamp` 에 찍는다
2. **처음 카드값으로** 분석: `analyze data/snapshots/1006_lamp --card data/snapshots/1006_desk/card.json`
   → `[조명] ΔE ...` 경고와 착색비율이 얼마나 흔들리는지 기록
3. 새 카드값으로 다시: `card data/snapshots/1006_lamp` → `analyze data/snapshots/1006_lamp`
   → 원래 값으로 돌아오는지 확인 (회색 카드 보정의 효과 = 발표 근거)

## 9-1. Piper Studio 를 켠 채로 판정 (대회 연결 경로 확인)

1~9절은 Piper Studio 를 끄고 D435 를 직접 열었다. 대회에서는 Piper Studio 가 켜진 상태에서 판정하므로,
**같은 사과로 이 경로가 같은 결과를 내는지** 마지막에 확인한다.

```bash
systemctl --user start piper-rsd               # Piper Studio 카메라 다시 켜기
# Piper Studio 화면: D435 켜기 → 회색 카드 보정 → 카메라 프로파일 저장
#                    카메라 설정에서 깊이 인코딩 창(near/far) 값 확인
python tools/judge_live.py --near 150 --far 1200 --card-roi <카드 x,y,w,h>
```

- 창에 사과마다 착색비율·등급이 실시간으로 뜬다. **s** = 현재 장면 저장(`data/live/`), **q** = 종료
- 5절에서 직접 찍은 결과와 등급이 같으면 대회 연결 경로가 검증된 것이다
- 다르면 `data/live/` 스냅샷을 `color_session.py analyze data/live` 로 다시 분석해 원인(깊이 정렬, 색 보정)을 찾는다
- 시작할 때 `깊이→컬러 보정 파일이 없다` 가 나오면 2절의 `--save-calib` 을 먼저 한다

## 10. 정리

- [ ] `data/snapshots/…`, `data/truth/…` 폴더를 공유 드라이브 `snapshots/` 에 업로드 (역할 A도 검출 학습에 쓴다)
- [ ] `configs/varieties.yaml` 의 `hue_range_deg`, `top_view_error`, `delta_up` 을 바꿔 PR (`[임시]` 표시 지우기)
- [ ] `_analysis/reject_curve.png` 와 오버레이 몇 장을 공유 드라이브 `videos/`(발표 자료)에 올리기
- [ ] `python tools/make_figures.py` → `data/figures/` 의 설계 근거 그림 6장도 함께 (기울기 보정, 커버리지, 회색 카드, 리젝트 곡선, 그림자, 다면 검사)
- [ ] Notion 실험 로그: 사과 수, 제안값, 일치율(⑤), 조명 실험 ΔE
- [ ] `systemctl --user start piper-rsd`
