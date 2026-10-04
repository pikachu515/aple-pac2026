# A+PLE — PAC 2026 분과① 안동청과

사과 착색 등급 판정 및 등급별 적재 시스템 (WeGo PIPER + Piper Studio)

> 설계 근거와 세부 전략은 팀 Notion의 **전략 종합** 문서를 기준으로 한다. 이 저장소는 코드와 설정만 관리한다.

## 시스템 구조

인식 · 판단 · 실행을 분리한다. 정책(ACT)은 "지정된 사과를 집어 지정된 칸에 놓는" 단일 동작만 수행하고, 검출 · 측정 · 등급 판정 · 배치 결정은 모두 결정론적 모듈이 처리한다.

```
① 상부 스냅샷 (D435 컬러+깊이)
② 검출 (YOLO)                      ┐
③ 기하 측정 (깊이)                  ├ 역할 A  aple/vision
⑦ 점유 격자 (상자 적재 높이)         ┘
④ 색 기준 고정·감시                 ┐
⑤ 착색비율 (색상각 + 면적 가중 + 다뷰)├ 역할 C  aple/grading
⑥ 등급 · 리젝트                     ┘
⑦ 파지 파라미터 적용                 ┐
⑧ 태스크 문장                       ├ 역할 B  aple/control
⑨ ACT 정책 실행 (Piper Studio)       ┘
   그리퍼 · 어댑터 · 힘 보정           ─ 역할 D  hardware/
⑩ 결과 기록 · 리셋                   ─ 공동    aple/orchestrator
```

## 폴더 구성

| 경로 | 담당 | 내용 |
|---|---|---|
| `aple/schema.py` | 공동 | **모듈 간 데이터 계약** — 변경 시 전원 합의 |
| `aple/vision/` | 역할 A | 검출, 깊이 기하 측정, 카메라–로봇 변환, 점유 격자 |
| `aple/grading/` | 역할 C | 색 기준 정규화, 착색비율, 다뷰 합산, 등급·리젝트 |
| `aple/control/` | 역할 B | 크기 구간 파라미터 적용, 태스크 문장, 그리퍼 토크 |
| `aple/orchestrator/` | 공동 (통합 담당) | 캡처 → 판단 → 태스크 지정 → 완료 대기 → 평가 → 리셋 |
| `hardware/` | 역할 D | 핑거 어댑터 CAD, FSR 보정 회로·스크립트 |
| `configs/` | 각 담당 | 현장에서 측정해 채우는 값 (임계값, 크기 구간, 변환 행렬) |
| `tools/` | 공동 | 장비 점검·촬영 보조 스크립트 |
| `tests/` | 공동 | 계약·계산 단위 시험 |
| `data/` | — | 로컬 데이터 위치 (git 제외) |

## 약속

**좌표계와 단위**
- 위치는 **로봇 베이스 좌표계, mm** 로 주고받는다
- 카메라–로봇 변환은 역할 A가 산출해 `configs/layout.yaml` 에 기록한다

**설정값**
- 현장에서 바뀌는 값은 코드에 상수로 넣지 않고 `configs/*.yaml` 에 둔다 (원칙 1 · 미지 파라미터 내성)

**브랜치**
- `main` 은 항상 실행 가능한 상태로 유지한다
- 작업은 `역할/주제` 브랜치에서 한다 — 예: `vision/depth-diameter`, `grading/hue-angle`
- `aple/schema.py` 를 바꾸는 변경은 Pull Request로 올리고 관련 역할의 확인을 받는다
- 대회 중에는 시간 관계상 `main` 직접 커밋을 허용하되, 커밋 전에 `pytest` 를 통과시킨다

**올리지 않는 것**
- 데이터셋, 녹화 영상, 학습 가중치(`*.pt`), 체크포인트 → 용량이 크다. 공유 드라이브에 둔다
- API 키 · 토큰 → `.env` 에만 두고 절대 커밋하지 않는다

## 환경

- Ubuntu 24.04, Piper Studio (버전 고정 — `docs/environment.md` 에 기록)
- Python 3.12

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## 장비 점검

```bash
python tools/rs_check.py          # D435 연결 · USB 3 협상 · 스트림 · 깊이 유효율
```

Piper Studio가 D435를 사용 중이면 장치가 점유되어 열리지 않는다. 점검은 Piper Studio 카메라 데몬을 멈춘 상태에서 수행한다.
