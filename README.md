# A+PLE — PAC 2026 분과① 안동청과

사과 착색 등급 판정 및 등급별 적재 시스템 (WeGo PIPER + Piper Studio)

> 논리 설계는 팀 Notion 「아키텍처 및 노드 설계」, 대회 운영 기준은 **전략 종합** 문서를 따른다. 이 저장소는 코드와 설정만 관리한다.
> 설계서 노드 ↔ 코드 대응과 대회용 조정 사항은 **[docs/architecture.md](docs/architecture.md)** 에 있다.

## 시스템 구조

인식 · 판단 · 실행을 분리한다. 검출 · 측정 · 등급 판정 · 배치 결정은 결정론적 모듈이 처리하고, 로봇 실행은 스크립트(IK)와 ACT 두 경로를 둔다. 로봇·카메라 연결은 Piper Studio가 맡는다.

```
M0 공통 인프라      Piper Studio 연동 (스냅샷 · 조명 상태 · 관절 목표 · ACT 실행)   aple/hal            역할 A·B
M1 인식 · 3D 위치   검출 → 구 피팅(중심·직경) → 상자 점유 격자                        aple/perception     역할 A
M2 색상 품질 판별   D435 탑뷰 착색비율(색상각 + 면적 가중) → 등급 · 리젝트             aple/inspection     역할 C
M5 판단 · 계획      크기 구간 파지 프로파일 → 작업 순서 → 적재 칸 · 고도               aple/planning       역할 B·D·A
M3 로봇 실행        Phase A (스크립트 / ACT) → Phase B 감속 하강 (항상 스크립트)       aple/execution      역할 B·D
M4 공정 관리        상태 기계 · 예외 처리 · 사이클 기록                               aple/orchestration  통합 담당
하드웨어            핑거 어댑터 · 패드 · FSR 보정                                     hardware/           역할 D
```

모듈 번호는 역할 분담 기준이다 (모듈 2 = 색상, 모듈 3 = 로봇).

## 폴더 구성

| 경로 | 담당 | 내용 |
|---|---|---|
| `aple/schema.py` | 공동 | **모듈 간 데이터 계약** (설계서 메시지 정의) — 변경 시 전원 합의 |
| `docs/interfaces.md` | 자동 생성 | 데이터 계약 다이어그램 — **[보기](docs/interfaces.md)** |
| `docs/architecture.md` | 공동 | 설계서 노드 ↔ 코드 대응, 필수 / 확장 / 대회 후 구분 |
| `aple/hal/` | 역할 A·B | Piper Studio 연동, 카메라–로봇 캘리브레이션 |
| `aple/perception/` | 역할 A | 검출, 3D 위치·직경, 상자 점유·적재 검증 |
| `aple/inspection/` | 역할 C | 착색비율, 다면 누적(모드 B), 등급·리젝트 |
| `aple/planning/` | 역할 B·D·A | 파지 프로파일, 작업 순서, 적재 계획 |
| `aple/execution/` | 역할 B·D | 집기·놓기 실행, 그리퍼 파지력 |
| `aple/orchestration/` | 통합 담당 | 상태 기계, 사이클 기록 |
| `hardware/` | 역할 D | 핑거 어댑터 CAD, FSR 보정 회로·스크립트 |
| `configs/` | 각 담당 | 현장에서 측정해 채우는 값 (설계서 9.3 설정 파일) |
| `tools/` | 공동 | 장비 점검, 다이어그램 생성 |
| `tests/` | 공동 | 계약·설정 단위 시험 |
| `data/` | — | 로컬 데이터 위치 (git 제외) |

## 약속

**좌표계와 단위**
- 위치는 **base_link(로봇 베이스) 좌표계**, 단위는 **SI (m, N, s, rad)** — mm는 설정 파일 키(`_mm`)에만
- 카메라–로봇 변환은 역할 A가 산출해 `configs/workcell.yaml` 에 기록한다
- 등급과 상자는 같은 번호 (UPPER=0, MIDDLE=1, REJECT=2)
- 위치·등급·파지 데이터는 같은 `scene_id` 끼리만 결합한다

**설정값**
- 현장에서 바뀌는 값은 코드에 상수로 넣지 않고 `configs/*.yaml` 에 둔다 (원칙 1 · 미지 파라미터 내성)

**브랜치**
- `main` 은 항상 실행 가능한 상태로 유지한다
- 작업은 `역할/주제` 브랜치에서 한다 — 예: `perception/sphere-fit`, `inspection/hue-angle`
- `aple/schema.py` 를 바꾸는 변경은 Pull Request로 올리고 관련 역할의 확인을 받는다
- `schema.py` 를 고친 뒤 `python tools/gen_interfaces.py` 로 다이어그램을 다시 만들어 함께 커밋한다 (다이어그램 직접 수정 금지)
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
