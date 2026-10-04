# data/

로컬 데이터 위치. 이 폴더의 내용은 git에 올라가지 않는다.

## 공유 드라이브

**https://drive.google.com/drive/folders/1AC4T7JOpZ2_NCyH4SxnhHdWYse2LCro4** (팀원 편집자 공유, 일반 액세스 제한)

| 폴더 | 내용 | 로컬 위치 |
|---|---|---|
| `snapshots/` | D435 스냅샷 (`tools/capture_snapshot.py` 결과). 폴더명 `날짜_장소` (예: `1005_desk`) | `data/snapshots/` |
| `yolo/` | YOLO 데이터셋, 학습 가중치(`*.pt`) | `data/yolo/` |
| `act/` | ACT 녹화 데이터셋, 체크포인트 | Piper Studio 데이터 폴더 |
| `logs/` | 사이클 기록 CSV, 실험 결과 | `logs/` |
| `videos/` | 시연 영상, 발표 자료용 | — |

- 업로드한 파일은 **올린 사람의 Google 용량**을 쓴다. ACT 녹화·체크포인트처럼 큰 파일은 학습 노트북 담당이 올린다
- 대회장 인터넷에 대비해 외장 SSD에도 같은 구조로 백업한다
