# 모듈 간 인터페이스

> 이 파일은 `tools/gen_interfaces.py` 가 `aple/schema.py` 로부터 자동 생성한다. **직접 고치지 않는다.**
> 인터페이스를 바꾸려면 `aple/schema.py` 의 클래스와 `FLOWS` 표를 고친다.

## 데이터 흐름

```mermaid
flowchart LR
    A["역할 A<br/>비전 타깃 인식 · 3D 위치"]
    C["역할 C<br/>다면 촬영 · 색상 품질 판별"]
    D["역할 D<br/>하드웨어 고도화"]
    B["역할 B<br/>로봇 파지 · 손상 방지 제어"]
    S["공동<br/>오케스트레이터 · 평가"]
    A -->|AppleObservation| C
    A -->|AppleGeometry| B
    A -->|BoxState| B
    C -->|GradeResult| B
    D -->|GraspParams| B
    B -->|CycleLog| S
```

| 보내는 역할 | 받는 역할 | 데이터 | 필드 |
|---|---|---|---|
| 역할 A · 비전 타깃 인식 · 3D 위치 | 역할 C · 다면 촬영 · 색상 품질 판별 | `AppleObservation` | `id`, `bbox`, `score`, `mask`, `normal_z`, `front_roi`, `depth_valid_ratio` |
| 역할 A · 비전 타깃 인식 · 3D 위치 | 역할 B · 로봇 파지 · 손상 방지 제어 | `AppleGeometry` | `id`, `xyz_mm`, `diameter_mm`, `size_bin` |
| 역할 A · 비전 타깃 인식 · 3D 위치 | 역할 B · 로봇 파지 · 손상 방지 제어 | `BoxState` | `box_id`, `wall_top_mm`, `cell_heights_mm`, `free_cells` |
| 역할 C · 다면 촬영 · 색상 품질 판별 | 역할 B · 로봇 파지 · 손상 방지 제어 | `GradeResult` | `id`, `grade`, `ratio_pct`, `ratio_by_view`, `reject_reason` |
| 역할 D · 하드웨어 고도화 | 역할 B · 로봇 파지 · 손상 방지 제어 | `GraspParams` | `size_bin`, `open_width_mm`, `grasp_height_mm`, `effort` |
| 역할 B · 로봇 파지 · 손상 방지 제어 | 공동 · 오케스트레이터 · 평가 | `CycleLog` | `cycle`, `plan`, `success`, `fail_code`, `cycle_time_s`, `note` |

## 데이터 구조

```mermaid
classDiagram
    direction LR
    class AppleObservation {
        +int id
        +tuple~int,int,int,int~ bbox
        +float score
        +ndarray mask
        +ndarray? normal_z
        +tuple~int,int,int,int~? front_roi
        +float depth_valid_ratio
    }
    class AppleGeometry {
        +int id
        +tuple~float,float,float~ xyz_mm
        +float diameter_mm
        +str size_bin
    }
    class BoxState {
        +str box_id
        +float wall_top_mm
        +list~float~ cell_heights_mm
        +list~int~ free_cells
    }
    class GradeResult {
        +int id
        +Grade grade
        +float ratio_pct
        +dict~str,float~ ratio_by_view
        +RejectReason reject_reason
    }
    class GraspParams {
        +str size_bin
        +float open_width_mm
        +float grasp_height_mm
        +int effort
    }
    class PlacePlan {
        +int apple_id
        +str box_id
        +int cell
        +float descend_to_mm
        +GraspParams grasp
        +str task_text
    }
    class CycleLog {
        +int cycle
        +PlacePlan? plan
        +bool success
        +FailCode fail_code
        +float cycle_time_s
        +str note
    }
    class Grade {
        <<enumeration>>
        HIGH
        MID
        REJECT
    }
    class RejectReason {
        <<enumeration>>
        NONE
        NEAR_BOUNDARY
        VIEW_DISAGREE
        OVERSIZE
    }
    class FailCode {
        <<enumeration>>
        NONE
        D1
        D2
        G1
        C1
        C2
        P1
        F1
        F2
        F3
        F4
    }
    GradeResult --> Grade : grade
    GradeResult --> RejectReason : reject_reason
    PlacePlan --> GraspParams : grasp
    CycleLog --> PlacePlan : plan
    CycleLog --> FailCode : fail_code
```
