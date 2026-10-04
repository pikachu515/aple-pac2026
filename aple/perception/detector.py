"""사과 검출 + 후처리 (설계서 apple_detector_yolo, detection_postprocess). 필수.

- Piper Studio YOLO 루프로 학습한 경계상자 모델, 또는 외부 학습 YOLO-seg 가중치
- 경계상자 모델이면 상자 안쪽을 깊이(트레이 면보다 솟은 영역)로 분리해 마스크 생성
- 트레이 ROI 안쪽만 사용, 트레이 면과 높이 차가 없는 검출은 오검출로 제외
"""

from __future__ import annotations

from aple.schema import AppleDetection2D, SceneSnapshot


def detect(snap: SceneSnapshot, cfg: dict) -> list[AppleDetection2D]:
    raise NotImplementedError
