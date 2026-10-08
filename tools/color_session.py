"""역할 C 실측 도구 — D435 스냅샷으로 색 보정·착색비율·등급 판정을 돌리고 기준값을 정한다.

순서 (docs/inspection_bench.md 참고)
  1. card     회색 카드 영역을 지정해 색 보정값(card.json)을 만든다
     patches  (선택) 적·황·백 레퍼런스 패치 영역을 지정해 기준 색(patch.json)을 만든다 — 조명이 틀어질 때만 쓰는 보정
  2. analyze  사과를 찾아 착색비율·등급을 계산하고 오버레이 이미지·CSV를 남긴다
  3. suggest  사과 픽셀 색상각 분포로 착색 판정 경계(hue)를 제안한다
  4. truth    사과마다 자세를 바꿔 찍은 6장으로 전표면 비율을 구하고, 탑뷰와의 오차를 잰다
  5. compare  육안 판정표와 비교해 일치율을 본다
  6. curve    판정 여유(delta_up)별 리젝트율 vs 남은 오분류율 곡선 → delta_up 제안, 발표 그래프
  7. repeat   같은 장면을 여러 번 찍은 폴더로 판정 재현성(사과별 착색비율 흔들림, 등급 일치율)을 잰다

  suggest · truth · curve 에 --apply 를 붙이면 제안값을 configs/varieties.yaml 에 바로 써 넣는다

    python tools/color_session.py card    data/snapshots/1006_desk
    python tools/color_session.py patches data/snapshots/1006_desk
    python tools/color_session.py analyze data/snapshots/1006_desk
    python tools/color_session.py suggest data/snapshots/1006_desk
    python tools/color_session.py truth   data/truth --card data/snapshots/1006_desk/card.json
    python tools/color_session.py compare data/truth/_analysis/truth.csv --labels data/truth/labels.csv
    python tools/color_session.py curve   data/truth/_analysis/truth.csv [--labels data/truth/labels.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aple.config import CONFIG_DIR, load_config, set_config_value  # noqa: E402
from aple.inspection.color import (CardCalib, apply_gain, bgr_to_lab, calibrate_card, card_brightness,  # noqa: E402
                                   hue_chroma, lighting_status, suggest_hue_max)
from aple.inspection.coloration import (InspectParams, measure_view, params_from_config, ratio_from_view,  # noqa: E402
                                        standard_grade)
from aple.inspection.evaluation import pick_delta_up, plot_curve, reject_curve  # noqa: E402
from aple.inspection.grade_classifier import classify  # noqa: E402
from aple.inspection.ref_patch import PatchCalib, calibrate_patches  # noqa: E402
from aple.inspection.standalone import find_apples  # noqa: E402
from aple.schema import Grade, LightState  # noqa: E402

GRADE_KO = {Grade.UPPER: "상", Grade.MIDDLE: "중", Grade.REJECT: "리젝트"}
VARIETIES = CONFIG_DIR / "varieties.yaml"
LABEL_MAP = {"상": Grade.UPPER, "u": Grade.UPPER, "upper": Grade.UPPER,
             "중": Grade.MIDDLE, "m": Grade.MIDDLE, "middle": Grade.MIDDLE}


# ---------------------------------------------------------------- 공통

# 사과 검출 가중치 — main 의 --weights (예: data/models/yolo11n.pt — COCO 면 apple 만). 없으면 깊이·모양만
WEIGHTS: Path | None = None


def load_snaps(folder: Path) -> list[tuple[Path, np.ndarray, np.ndarray, tuple]]:
    out = []
    for p in sorted(folder.glob("*.npz")):
        with np.load(p) as z:
            out.append((p, z["color"], z["depth_m"], tuple(float(v) for v in z["intrinsics"])))
    if not out:
        raise SystemExit(f"{folder} 에 .npz 스냅샷이 없다 - tools/capture_snapshot.py 로 먼저 찍는다")
    return out


def analysis_dir(folder: Path) -> Path:
    d = folder / "_analysis"
    d.mkdir(exist_ok=True)
    return d


def find_card(folder: Path, card_arg: str | None) -> CardCalib | None:
    path = Path(card_arg) if card_arg else folder / "card.json"
    if not path.exists():
        print(f"[알림] 색 보정 파일 없음 ({path}) - 보정 없이 원본 색으로 계산한다")
        return None
    return CardCalib.from_dict(json.loads(path.read_text(encoding="utf-8")))


def params(variety: str | None) -> InspectParams:
    return params_from_config(load_config(), variety)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def overlay(color: np.ndarray, items: list[tuple[int, object, str]]) -> np.ndarray:
    """착색 픽셀은 빨강, 판정 대상 비착색 픽셀은 파랑으로 칠하고 번호·비율·등급을 쓴다."""
    img = color.copy()
    for idx, vm, label in items:
        if vm is None:
            continue
        img[vm.valid & ~vm.colored] = (0.5 * img[vm.valid & ~vm.colored] + [127, 40, 0]).astype(np.uint8)
        img[vm.colored] = (0.5 * img[vm.colored] + [0, 0, 127]).astype(np.uint8)
        ys, xs = np.nonzero(vm.valid)
        cv2.putText(img, f"#{idx} {label}", (int(xs.min()), max(int(ys.min()) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, f"#{idx} {label}", (int(xs.min()), max(int(ys.min()) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


GRADE_EN = {Grade.UPPER: "UPPER", Grade.MIDDLE: "MIDDLE", Grade.REJECT: "REJECT"}


# ---------------------------------------------------------------- 1. card

def cmd_card(folder: Path, roi_arg: str | None, file_arg: str | None) -> CardCalib:
    snaps = load_snaps(folder)
    path, color = (Path(file_arg), np.load(file_arg)["color"]) if file_arg else (snaps[0][0], snaps[0][1])
    if roi_arg:
        roi = tuple(int(v) for v in roi_arg.split(","))
    else:
        r = cv2.selectROI("drag gray card, then ENTER", color, showCrosshair=False)
        cv2.destroyAllWindows()
        roi = tuple(int(v) for v in r)
    calib = calibrate_card(color, roi)
    (folder / "card.json").write_text(json.dumps(calib.to_dict(), indent=2), encoding="utf-8")
    print(f"기준 사진 {path.name}, 카드 영역 {roi}")
    print(f"카드 밝기 {card_brightness(color, roi):.0f} (118 근처가 좋다. 200 이상이면 노출 과다, 60 이하면 부족)")
    print(f"채널 게인 R {calib.rgb_gain[0]:.2f}  G {calib.rgb_gain[1]:.2f}  B {calib.rgb_gain[2]:.2f}")
    print(f"저장: {folder / 'card.json'}")
    return calib


PATCH_NAMES = ("red", "yellow", "white")


def cmd_patches(folder: Path, roi_args: list[str] | None, file_arg: str | None) -> PatchCalib:
    """레퍼런스 패치 기준 색 → folder/patch.json. 카드(card.json)가 있으면 4번째 패치로 함께 넣는다."""
    snaps = load_snaps(folder)
    path, color = (Path(file_arg), np.load(file_arg)["color"]) if file_arg else (snaps[0][0], snaps[0][1])
    rois: dict[str, tuple[int, int, int, int]] = {}
    if roi_args:
        for item in roi_args:
            name, _, box = item.partition("=")
            rois[name] = tuple(int(v) for v in box.split(","))
    else:
        for name in PATCH_NAMES:
            r = cv2.selectROI(f"drag {name} patch, then ENTER", color, showCrosshair=False)
            cv2.destroyAllWindows()
            rois[name] = tuple(int(v) for v in r)
    card = find_card(folder, None)
    if card is not None and "card" not in rois:
        rois["card"] = tuple(card.roi)
    calib = calibrate_patches(color, rois)
    (folder / "patch.json").write_text(json.dumps(calib.to_dict(), indent=2), encoding="utf-8")
    print(f"기준 사진 {path.name}")
    for name, lin in calib.ref_lin.items():
        print(f"  {name:7s} 영역 {calib.rois[name]}  선형 RGB {lin[0]:.3f} {lin[1]:.3f} {lin[2]:.3f}")
    print(f"저장: {folder / 'patch.json'}  (configs/color.yaml 의 color.patch_json 에 이 경로를 넣는다)")
    return calib


# ---------------------------------------------------------------- 2. analyze

def analyze_folder(folder: Path, calib: CardCalib | None, p: InspectParams, save_images: bool = True) -> list[dict]:
    out_dir = analysis_dir(folder)
    rows = []
    for path, raw, depth, intr in load_snaps(folder):
        color = apply_gain(raw, calib)
        light = lighting_status(raw, calib) if calib else None
        if light is not None and light.light_state != LightState.OK:
            print(f"[조명] {path.name}: 카드 색 변화 ΔE {light.delta_e:.1f} → {light.light_state.name}")
        try:
            dets = find_apples(color, depth, intr, weights=WEIGHTS)
        except ValueError as e:
            print(f"[건너뜀] {path.name}: {e}")
            continue
        items = []
        for det in dets:
            vm = measure_view(color, depth, det.mask, intr, p)
            if vm is None:
                items.append((det.det_id, None, "측정 실패"))
                continue
            shaded = vm.shadow_fraction > p.max_shadow_fraction        # 그림자가 많으면 판정 보류 (재촬영)
            g = classify(None if shaded else ratio_from_view(vm, p, 0, det.det_id), p, light=light, apple_id=det.det_id)
            items.append((det.det_id, vm, f"{vm.ratio_visible * 100:.0f}% {GRADE_EN[g.grade]}"))
            rows.append({
                "file": path.name, "apple": det.det_id,
                "diameter_mm": round(vm.sphere.radius * 2000, 1),
                "ratio": round(vm.ratio_visible, 3), "ratio_lo": round(g.ratio_lo, 3), "ratio_hi": round(g.ratio_hi, 3),
                "coverage": round(vm.coverage, 3), "grade": GRADE_KO[g.grade], "reason": g.reject_reason.name,
                "standard": standard_grade(vm.ratio_visible, p), "shadow_pct": round(vm.shadow_fraction * 100, 1),
                "depth_fit_mm": round(vm.sphere.fit_residual * 1000, 1),
                "light_dE": round(light.delta_e, 2) if light else "",
            })
        if save_images:
            cv2.imwrite(str(out_dir / f"{path.stem}_overlay.png"), overlay(color, items))
    write_csv(out_dir / "results.csv", rows)
    return rows


def cmd_analyze(folder: Path, card: str | None, variety: str | None) -> None:
    p = params(variety)
    rows = analyze_folder(folder, find_card(folder, card), p)
    print()
    print(f"품종 {p.variety}, 상자 기준 '{p.box_boundary}' 이상 {p.coloring_threshold:.0%}, 착색 색상각 {p.hue_range_deg}, "
          f"판정 구간 = 탑뷰 {-p.top_view_bias * 100:+.0f}%p ± {p.top_view_error * 100:.0f}%p")
    print(f"{'파일':<16}{'번호':>4}{'직경mm':>8}{'착색':>7}{'규격':>5}{'판정':>7}  사유")
    for r in rows:
        shade = f" (그림자 {r['shadow_pct']:.0f}%)" if r["reason"] == "OCCLUDED" else ""
        print(f"{r['file']:<16}{r['apple']:>4}{r['diameter_mm']:>8}{r['ratio'] * 100:>6.0f}%{r['standard']:>5}"
              f"{r['grade']:>7}  {r['reason']}{shade}")
    print(f"\n오버레이·CSV: {analysis_dir(folder)}  (빨강 = 착색, 파랑 = 비착색, 칠해지지 않은 가장자리는 판정 제외)")


# ---------------------------------------------------------------- 3. suggest

def collect_apple_pixels(folder: Path, calib: CardCalib | None, p: InspectParams) -> tuple[np.ndarray, np.ndarray]:
    hs, cs = [], []
    for path, raw, depth, intr in load_snaps(folder):
        color = apply_gain(raw, calib)
        for det in find_apples(color, depth, intr, weights=WEIGHTS):
            vm = measure_view(color, depth, det.mask, intr, p)
            if vm is None:
                continue
            h, c = hue_chroma(bgr_to_lab(color[vm.valid]))
            hs.append(h)
            cs.append(c)
    if not hs:
        raise SystemExit("사과를 찾지 못했다 - analyze 결과 오버레이로 분할부터 확인한다")
    return np.concatenate(hs), np.concatenate(cs)


def cmd_suggest(folder: Path, card: str | None, variety: str | None, apply_to: Path | None = None) -> float:
    p = params(variety)
    h, c = collect_apple_pixels(folder, find_card(folder, card), p)
    th = suggest_hue_max(h, c, p.min_chroma)
    use = c >= p.min_chroma
    red, rest = (h[use] <= th).mean(), (h[use] > th).mean()
    print(f"사과 픽셀 {use.sum():,}개 (채도 {p.min_chroma} 이상)")
    print(f"제안 경계 색상각: {th:.1f}°  →  configs/varieties.yaml 의 hue_range_deg: [{p.hue_range_deg[0]:.0f}, {th:.0f}]")
    print(f"  경계 아래(착색) {red:.0%} / 위(비착색) {rest:.0%}")
    if min(red, rest) < 0.1:
        print("  [주의] 한쪽 무리가 10% 미만 - 빨간 사과와 노란·초록 사과를 섞어 찍어야 경계가 의미 있다")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(h[use], bins=120, range=(-60, 150), color="#888")
        ax.axvline(th, color="r", label=f"suggested {th:.0f} deg")
        ax.axvline(p.hue_range_deg[1], color="b", ls="--", label=f"current {p.hue_range_deg[1]:.0f} deg")
        ax.set_xlabel("hue angle h (deg)  0 = red, 90 = yellow")
        ax.legend()
        fig.tight_layout()
        fig.savefig(analysis_dir(folder) / "hue_hist.png", dpi=120)
        print(f"  히스토그램: {analysis_dir(folder) / 'hue_hist.png'}")
    except ImportError:
        pass
    if apply_to is not None:
        lo = p.hue_range_deg[0]
        print("  적용:", set_config_value(apply_to, f"varieties.{p.variety}.hue_range_deg", [int(lo), int(round(th))]).strip())
    return th


# ---------------------------------------------------------------- 4. truth

def truth_root(root: Path, calib: CardCalib | None, p: InspectParams) -> tuple[list[dict], dict]:
    """root/<사과>/snap_*.npz — 첫 장은 자연스럽게 놓인 자세(탑뷰 판정에 쓰는 장), 나머지는 굴려서 다른 면이 위로.

    자세마다 위를 향한 면이 다르므로 **모든 장이 '탑뷰 하나'의 표본**이다. 사과 5개 × 6자세면 30개 표본으로
    탑뷰 오차를 잰다 (summary 는 전체 자세 기준). 같은 사과의 자세끼리는 서로 독립이 아니라는 점은 감안한다.
    반환 rows 는 사과별(첫 장 = 평소 자세), summary["views"] 는 자세별 행이다.
    """
    rows, view_rows = [], []
    for d in sorted(x for x in root.iterdir() if x.is_dir() and not x.name.startswith("_")):
        views = []
        for path, raw, depth, intr in load_snaps(d):
            color = apply_gain(raw, calib)
            dets = find_apples(color, depth, intr, weights=WEIGHTS)
            if not dets:
                print(f"[건너뜀] {d.name}/{path.name}: 사과를 찾지 못함")
                continue
            det = max(dets, key=lambda x: x.mask.sum())       # 사과 1개만 놓고 찍는다
            vm = measure_view(color, depth, det.mask, intr, p)
            if vm is not None:
                views.append(vm)
        if len(views) < 2:
            print(f"[건너뜀] {d.name}: 유효 뷰 {len(views)}장")
            continue
        cov = np.array([v.coverage for v in views])
        rat = np.array([v.ratio_visible for v in views])
        full = float((rat * cov).sum() / cov.sum())          # 여러 자세의 관측 면을 넓이로 가중 평균
        rows.append({"apple": d.name, "views": len(views), "top_ratio": round(float(rat[0]), 3),
                     "full_ratio": round(full, 3), "diff": round(float(rat[0]) - full, 3),
                     "diameter_mm": round(float(np.median([v.sphere.radius for v in views])) * 2000, 1)})
        view_rows += [{"apple": d.name, "view": i, "top_ratio": round(float(r), 3), "full_ratio": round(full, 3),
                       "diff": round(float(r) - full, 3)} for i, r in enumerate(rat)]
    if not rows:
        raise SystemExit("처리한 사과가 없다")
    diffs = np.abs([r["diff"] for r in view_rows])
    p90 = float(np.percentile(diffs, 90))
    summary = {"apples": len(rows), "samples": len(view_rows), "mae": float(diffs.mean()), "p90": p90,
               "views": view_rows,
               "all": _bias_and_error([r["diff"] for r in view_rows]),               # 모든 자세 (방향이 무작위일 때)
               "first": _bias_and_error([r["diff"] for r in view_rows if r["view"] == 0])}   # 평소 자세만 (꼭지가 위)
    summary["suggested_top_view_bias"], summary["suggested_top_view_error"] = summary["all"]
    return rows, summary


def _bias_and_error(signed: list[float]) -> tuple[float, float]:
    """탑뷰 − 전표면 차이들 → (치우침 = 중앙값, 구간 반폭 = 치우침을 뺀 차이의 90% 지점).

    치우침을 빼지 않고 |차이| 로 대칭 구간을 잡으면, 한쪽으로 쏠린 오차를 덮느라 구간이 두 배 가까이 넓어진다.
    """
    d = np.asarray(signed, dtype=float)
    bias = float(np.median(d))
    resid = float(np.percentile(np.abs(d - bias), 90))
    return round(bias, 2), max(0.03, math.ceil(resid * 100) / 100)


def cmd_truth(root: Path, card: str | None, variety: str | None, apply_to: Path | None = None,
              poses: str = "all") -> dict:
    """poses: all = 모든 자세 표본 (사과 방향이 무작위일 때), first = 평소 자세(꼭지가 위)만 (현장에서 그렇게 놓일 때)."""
    p = params(variety)
    rows, s = truth_root(root, find_card(root, card), p)
    bias, err = s[poses]
    write_csv(analysis_dir(root) / "truth.csv", rows)
    write_csv(analysis_dir(root) / "truth_views.csv", s["views"])     # 자세별 표본 - curve 에도 그대로 쓸 수 있다
    print(f"{'사과':<12}{'뷰':>3}{'탑뷰':>8}{'전표면':>8}{'차이':>8}")
    for r in rows:
        print(f"{r['apple']:<12}{r['views']:>3}{r['top_ratio'] * 100:>7.0f}%{r['full_ratio'] * 100:>7.0f}%{r['diff'] * 100:>+7.0f}%p")
    print(f"\n사과 {s['apples']}개 x 자세 = 표본 {s['samples']}개 - 탑뷰 오차 평균 {s['mae'] * 100:.1f}%p, "
          f"90% 지점 {s['p90'] * 100:.1f}%p")
    for name, label in (("all", "모든 자세"), ("first", "평소 자세(꼭지 위)만")):
        b, e = s[name]
        print(f"  {label:<14} 치우침 {b * 100:+.0f}%p, 남는 흔들림 ±{e * 100:.0f}%p  → 판정 구간 = 탑뷰 {-b * 100:+.0f}%p ± {e * 100:.0f}%p")
    print(f"제안 ({'모든 자세' if poses == 'all' else '평소 자세'} 기준): top_view_bias: {bias:.2f}, top_view_error: {err:.2f}  "
          f"(현재 {p.top_view_bias:.2f}, {p.top_view_error:.2f})")
    print("  ※ 현장에서 사과가 놓이는 방향에 맞춰 고른다 — 꼭지가 위로 놓이면 --poses first")
    print(f"CSV: {analysis_dir(root) / 'truth.csv'} (사과별), truth_views.csv (자세별 - curve 에 넣으면 표본이 늘어난다)")
    if apply_to is not None:
        print("  적용:", set_config_value(apply_to, "inspection.top_view_bias", bias).strip())
        print("  적용:", set_config_value(apply_to, "inspection.top_view_error", err).strip())
    return s


# ---------------------------------------------------------------- 5. compare

def cmd_compare(truth_csv: Path, labels_csv: Path, variety: str | None) -> dict:
    """육안 판정표(apple,label — label: 상 / 중 / 애매)와 탑뷰·전표면 판정을 비교한다."""
    p = params(variety)
    T = p.coloring_threshold
    labels = {r["apple"].strip(): r["label"].strip().lower()
              for r in csv.DictReader(labels_csv.open(encoding="utf-8-sig"))}
    stats = {"n": 0, "top_agree": 0, "full_agree": 0, "top_reject": 0}
    print(f"{'사과':<12}{'육안':>6}{'탑뷰판정':>10}{'전표면':>8}")
    for r in csv.DictReader(truth_csv.open(encoding="utf-8-sig")):
        lab = labels.get(r["apple"])
        if lab is None:
            continue
        top = float(r["top_ratio"])
        g_top = classify(_cr(top, p), p).grade
        g_full = Grade.UPPER if float(r["full_ratio"]) >= T else Grade.MIDDLE
        print(f"{r['apple']:<12}{lab:>6}{GRADE_KO[g_top]:>10}{GRADE_KO[g_full]:>8}")
        want = LABEL_MAP.get(lab)
        if want is None:                     # 애매 — 일치율 계산에서 뺀다
            continue
        stats["n"] += 1
        stats["top_agree"] += g_top == want
        stats["top_reject"] += g_top == Grade.REJECT
        stats["full_agree"] += g_full == want
    if stats["n"]:
        n = stats["n"]
        print(f"\n육안 상/중 {n}개 기준 - 전표면 일치 {stats['full_agree'] / n:.0%}, "
              f"탑뷰 일치 {stats['top_agree'] / n:.0%} (리젝트로 보류 {stats['top_reject'] / n:.0%})")
    return stats


def _cr(ratio: float, p: InspectParams):
    from aple.inspection.coloration import top_view_interval
    from aple.schema import ColorRatio, InspectMode
    est, lo, hi = top_view_interval(ratio, p)
    return ColorRatio(0, 0, InspectMode.TOP_SINGLE, est, lo, hi, 0.25, 0.0, 1, p.variety)


# ---------------------------------------------------------------- 6. curve

def cmd_curve(truth_csv: Path, labels_csv: Path | None, variety: str | None, max_up_error: float = 0.0,
              apply_to: Path | None = None, margin: float = 0.03) -> dict:
    """정답: --labels 가 있으면 육안 판정(상/중, '애매' 제외), 없으면 전표면 착색비율 ≥ 기준."""
    p = params(variety)
    rows = list(csv.DictReader(truth_csv.open(encoding="utf-8-sig")))
    labels = ({r["apple"].strip(): r["label"].strip().lower()
               for r in csv.DictReader(labels_csv.open(encoding="utf-8-sig"))} if labels_csv else None)
    names, measured, truth = [], [], []
    for r in rows:
        if labels is not None:
            g = LABEL_MAP.get(labels.get(r["apple"], ""))
            if g is None:
                continue
        else:
            g = Grade.UPPER if float(r["full_ratio"]) >= p.coloring_threshold else Grade.MIDDLE
        names.append(r["apple"])
        measured.append(float(r["top_ratio"]))
        truth.append(g)
    if len(measured) < 3:
        raise SystemExit("비교할 사과가 3개 미만이다")

    curve = reject_curve(measured, truth, p)
    best = pick_delta_up(curve, max_up_error)
    out = analysis_dir(truth_csv.parent.parent if truth_csv.parent.name == "_analysis" else truth_csv.parent)
    write_csv(out / "curve.csv", [{"delta_up": c.delta_up, "reject_rate": round(c.reject_rate, 3),
                                   "up_error_rate": round(c.up_error_rate, 3),
                                   "down_error_rate": round(c.down_error_rate, 3)} for c in curve])
    try:
        plot_curve(curve, best, out / "reject_curve.png")
    except ImportError:
        pass
    src = "육안 판정" if labels is not None else "전표면 착색비율"
    print(f"정답 기준: {src}, 사과 {len(measured)}개 (상 {truth.count(Grade.UPPER)} / 중 {truth.count(Grade.MIDDLE)})")
    print(f"{'delta_up':>9}{'리젝트':>8}{'중→상':>8}{'상→중':>8}")
    for c in curve[::5]:
        print(f"{c.delta_up:>+9.2f}{c.reject_rate:>8.0%}{c.up_error_rate:>8.0%}{c.down_error_rate:>8.0%}")
    print()
    print(f"제안: delta_up {best.delta_up:+.2f} - 리젝트 {best.reject_rate:.0%}, '중→상' {best.up_error_rate:.0%}, "
          f"'상→중' {best.down_error_rate:.0%}  (현재 {p.delta_up:+.2f})")
    print(f"그래프·CSV: {out}")
    reachable = best.up_error_rate <= max_up_error + 1e-12
    if not reachable:
        stuck = [a for a, m, t in zip(names, measured, truth)
                 if t == Grade.MIDDLE and m - p.top_view_error >= p.coloring_threshold + curve[-1].delta_up]
        print(f"  [중요] 탑뷰만으로는 '중→상' 오류를 {max_up_error:.0%} 이하로 만들 수 없다.")
        print(f"         윗면만 빨갛고 나머지는 덜 익은 사과가 있다: {', '.join(stuck) or '-'}")
        print("         → 이런 사과는 다면 검사(모드 B, 적응형)로만 잡힌다. 발표 근거로 쓸 수 있다")
    if len(measured) < 15:
        print("  [주의] 사과 수가 적어 곡선이 거칠다 - 운용점은 참고용, 현장 표본으로 다시 확인")
    if apply_to is not None:
        value = round(best.delta_up + margin, 2)            # 표본이 적어 경계에 붙은 값 → 여유를 더한다
        print(f"  적용 (제안 {best.delta_up:+.2f} + 여유 {margin:.2f}):",
              set_config_value(apply_to, f"varieties.{p.variety}.delta_up", value).strip())
    return {"chosen": best, "curve": curve, "reachable": reachable}


# ---------------------------------------------------------------- 7. repeat

def cmd_repeat(folder: Path, card: str | None, variety: str | None) -> dict:
    """같은 장면(사과를 움직이지 않음)을 여러 장 찍은 폴더 → 사과별 착색비율 표준편차와 등급 일치율.

    사과 번호는 장마다 위→아래, 왼→오 순서로 매겨지므로 사과를 움직이지 않으면 같은 번호가 같은 사과다.
    """
    p = params(variety)
    rows = analyze_folder(folder, find_card(folder, card), p, save_images=False)
    by_apple: dict[int, list[dict]] = {}
    for r in rows:
        by_apple.setdefault(r["apple"], []).append(r)
    files = len({r["file"] for r in rows})
    out, summary = [], {"frames": files, "apples": {}}
    print(f"같은 장면 {files}장, 사과 {len(by_apple)}개")
    print(f"{'사과':>4}{'장':>4}{'평균':>8}{'표준편차':>9}{'범위':>8}{'등급 일치':>9}")
    for a, rs in sorted(by_apple.items()):
        ratios = np.array([r["ratio"] for r in rs])
        grades = [r["grade"] for r in rs]
        mode = max(set(grades), key=grades.count)
        agree = grades.count(mode) / len(grades)
        rec = {"apple": a, "n": len(rs), "mean": round(float(ratios.mean()), 4), "std": round(float(ratios.std()), 4),
               "range": round(float(ratios.max() - ratios.min()), 4), "grade": mode, "grade_agreement": round(agree, 3)}
        out.append(rec)
        summary["apples"][a] = rec
        print(f"{a:>4}{len(rs):>4}{rec['mean'] * 100:>7.1f}%{rec['std'] * 100:>8.2f}%p{rec['range'] * 100:>7.1f}%p"
              f"{agree:>9.0%}")
    if files < 2 or not out:
        print("  [주의] 같은 장면을 2장 이상 찍어야 재현성을 잴 수 있다")
    write_csv(analysis_dir(folder) / "repeat.csv", out)
    worst = max((r["std"] for r in out), default=0.0)
    summary["worst_std"] = worst
    print()
    print(f"가장 크게 흔들린 사과의 표준편차 {worst * 100:.2f}%p  (판정 구간 ±{p.top_view_error * 100:.0f}%p 와 비교)")
    return summary


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("card")
    a.add_argument("folder", type=Path)
    a.add_argument("--roi", help="x,y,w,h (화면 없이 지정할 때)")
    a.add_argument("--file", help="기준으로 쓸 스냅샷 (기본: 폴더 첫 장)")
    pa = sub.add_parser("patches")
    pa.add_argument("folder", type=Path)
    pa.add_argument("--roi", action="append", help="name=x,y,w,h (화면 없이 지정할 때, 패치마다 반복)")
    pa.add_argument("--file", help="기준으로 쓸 스냅샷 (기본: 폴더 첫 장)")
    for name in ("analyze", "suggest", "truth", "repeat"):
        b = sub.add_parser(name)
        b.add_argument("folder", type=Path)
        b.add_argument("--card", help="card.json 경로 (기본: 폴더 안 card.json)")
        b.add_argument("--variety", choices=["fuji", "hongro"])
        b.add_argument("--weights", type=Path, help="사과 검출 가중치 (예: data/models/yolo11n.pt). 없으면 깊이·모양만")
        if name in ("suggest", "truth"):
            b.add_argument("--apply", action="store_true", help="제안값을 configs/varieties.yaml 에 바로 써 넣는다")
        if name == "truth":
            b.add_argument("--poses", choices=["all", "first"], default="all",
                           help="all: 모든 자세 표본 (방향 무작위), first: 평소 자세(꼭지 위)만 (현장에서 그렇게 놓일 때)")
    c = sub.add_parser("compare")
    c.add_argument("truth_csv", type=Path)
    c.add_argument("--labels", type=Path, required=True)
    c.add_argument("--variety", choices=["fuji", "hongro"])
    k = sub.add_parser("curve")
    k.add_argument("truth_csv", type=Path)
    k.add_argument("--labels", type=Path)
    k.add_argument("--variety", choices=["fuji", "hongro"])
    k.add_argument("--max-up-error", type=float, default=0.0, help="허용할 '중→상' 오류 비율 (기본 0)")
    k.add_argument("--apply", action="store_true", help="제안 delta_up + 여유를 configs/varieties.yaml 에 써 넣는다")
    k.add_argument("--margin", type=float, default=0.03, help="--apply 때 제안값에 더할 여유 (기본 0.03)")
    args = ap.parse_args(argv)

    global WEIGHTS
    WEIGHTS = getattr(args, "weights", None)

    if args.cmd == "card":
        cmd_card(args.folder, args.roi, args.file)
    elif args.cmd == "patches":
        cmd_patches(args.folder, args.roi, args.file)
    elif args.cmd == "analyze":
        cmd_analyze(args.folder, args.card, args.variety)
    elif args.cmd == "suggest":
        cmd_suggest(args.folder, args.card, args.variety, VARIETIES if args.apply else None)
    elif args.cmd == "repeat":
        cmd_repeat(args.folder, args.card, args.variety)
    elif args.cmd == "truth":
        cmd_truth(args.folder, args.card, args.variety, VARIETIES if args.apply else None, args.poses)
    elif args.cmd == "compare":
        cmd_compare(args.truth_csv, args.labels, args.variety)
    else:
        cmd_curve(args.truth_csv, args.labels, args.variety, args.max_up_error,
                  VARIETIES if args.apply else None, args.margin)


if __name__ == "__main__":
    main()
