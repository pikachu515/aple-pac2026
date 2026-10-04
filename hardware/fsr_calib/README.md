# 토크–힘 보정 (S8-6)

1. FSR을 핑거 사이에 물리고 토크 설정 200~2000 구간에서 실제 힘(N) 측정
2. 측정값을 `measurements.csv` (effort, force_n, trial) 로 저장
3. 보정 곡선 → `configs/grasp_bins.yaml` 의 effort 값 갱신
