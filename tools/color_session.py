"""역할 C 실측 도구 — D435 스냅샷으로 색 보정·착색비율·등급 판정을 돌리고 기준값을 정한다.

순서 (docs/inspection_bench.md 참고)
  1. card     회색 카드 영역을 지정해 색 보정값(card.json)을 만든다
  2. analyze  사과를 찾아 착색비율·등급을 계산하고 오버레이 이미지·CSV를 남긴다
  3. suggest  사과 픽셀 색상각 분포로 착색 판정 경계(hue)를 제안한다
  4. truth    사과마다 자세를 바꿔 찍은 6장으로 전표면 비율을 구하고, 탑뷰와의 오차를 잰다
  5. compare  육안 판정표와 비교해 일치율을 본다

    python tools/color_session.py card    data/snapshots/1006_desk
    python tools/color_session.py analyze data/snapshots/1006_desk
    python tools/color_session.py suggest data/snapshots/1006_desk
    python tools/color_session.py truth   data/truth --card data/snapshots/1006_desk/card.json
    python tools/color_session.py compare data/truth/_analysis/truth.csv --labels data/truth/labels.csv
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
from aple.config import load_config  # noqa: E402
from aple.inspection.color import (CardCalib, apply_gain, bgr_to_lab, calibrate_card, card_brightness,  # noqa: E402
                                   hue_chroma, lighting_status, suggest_hue_max)
from aple.inspection.coloration import InspectParams, measure_view, params_from_config, ratio_from_view  # noqa: E402
from aple.inspection.grade_classifier import classify  # noqa: E402
from aple.inspection.standalone import segment_apples  # noqa: E402
from aple.schema import Grade, LightState  # noqa: E402

GRADE_KO = {Grade.UPPER: "상", Grade.MIDDLE: "중", Grade.REJECT: "리젝트"}
LABEL_MAP = {"상": Grade.UPPER, "u": Grade.UPPER, "upper": Grade.UPPER,
             "중": Grade.MIDDLE, "m": Grade.MIDDLE, "middle": Grade.MIDDLE}


# ---------------------------------------------------------------- 공통

def load_snaps(folder: Path) -> list[tuple[Path, np.ndarray, np.ndarray, tuple]]:
    out = []
    for p in sorted(folder.glob("*.npz")):
        with np.load(p) as z:
            out.append((p, z["color"], z["depth_m"], tuple(float(v) for v in z["intrinsics"])))
    if not out:
        raise SystemExit(f"{folder} 에 .npz 스냅샷이 없다 — tools/capture_snapshot.py 로 먼저 찍는다")
    return out


def analysis_dir(folder: Path) -> Path:
    d = folder / "_analysis"
    d.mkdir(exist_ok=True)
    return d


def find_card(folder: Path, card_arg: str | None) -> CardCalib | None:
    path = Path(card_arg) if card_arg else folder / "card.json"
    if not path.exists():
        print(f"[알림] 색 보정 파일 없음 ({path}) — 보정 없이 원본 색으로 계산한다")
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
            dets = segment_apples(color, depth, intr)
        except ValueError as e:
            print(f"[건너뜀] {path.name}: {e}")
            continue
        items = []
        for det in dets:
            vm = measure_view(color, depth, det.mask, intr, p)
            if vm is None:
                items.append((det.det_id, None, "측정 실패"))
                continue
            g = classify(ratio_from_view(vm, p, 0, det.det_id), p, light=light)
            items.append((det.det_id, vm, f"{vm.ratio_visible * 100:.0f}% {GRADE_EN[g.grade]}"))
            rows.append({
                "file": path.name, "apple": det.det_id,
                "diameter_mm": round(vm.sphere.radius * 2000, 1),
                "ratio": round(vm.ratio_visible, 3), "ratio_lo": round(g.ratio_lo, 3), "ratio_hi": round(g.ratio_hi, 3),
                "coverage": round(vm.coverage, 3), "grade": GRADE_KO[g.grade], "reason": g.reject_reason.name,
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
    print(f"\n품종 {p.variety}, 기준 {p.coloring_threshold:.0%}, 착색 색상각 {p.hue_range_deg}, 판정 구간 ±{p.top_view_error:.2f}")
    print(f"{'파일':<16}{'번호':>4}{'직경mm':>8}{'착색':>7}{'등급':>6}  사유")
    for r in rows:
        print(f"{r['file']:<16}{r['apple']:>4}{r['diameter_mm']:>8}{r['ratio'] * 100:>6.0f}%{r['grade']:>6}  {r['reason']}")
    print(f"\n오버레이·CSV: {analysis_dir(folder)}  (빨강 = 착색, 파랑 = 비착색, 칠해지지 않은 가장자리는 판정 제외)")


# ---------------------------------------------------------------- 3. suggest

def collect_apple_pixels(folder: Path, calib: CardCalib | None, p: InspectParams) -> tuple[np.ndarray, np.ndarray]:
    hs, cs = [], []
    for path, raw, depth, intr in load_snaps(folder):
        color = apply_gain(raw, calib)
        for det in segment_apples(color, depth, intr):
            vm = measure_view(color, depth, det.mask, intr, p)
            if vm is None:
                continue
            h, c = hue_chroma(bgr_to_lab(color[vm.valid]))
            hs.append(h)
            cs.append(c)
    if not hs:
        raise SystemExit("사과를 찾지 못했다 — analyze 결과 오버레이로 분할부터 확인한다")
    return np.concatenate(hs), np.concatenate(cs)


def cmd_suggest(folder: Path, card: str | None, variety: str | None) -> float:
    p = params(variety)
    h, c = collect_apple_pixels(folder, find_card(folder, card), p)
    th = suggest_hue_max(h, c, p.min_chroma)
    use = c >= p.min_chroma
    red, rest = (h[use] <= th).mean(), (h[use] > th).mean()
    print(f"사과 픽셀 {use.sum():,}개 (채도 {p.min_chroma} 이상)")
    print(f"제안 경계 색상각: {th:.1f}°  →  configs/varieties.yaml 의 hue_range_deg: [{p.hue_range_deg[0]:.0f}, {th:.0f}]")
    print(f"  경계 아래(착색) {red:.0%} / 위(비착색) {rest:.0%}")
    if min(red, rest) < 0.1:
        print("  [주의] 한쪽 무리가 10% 미만 — 빨간 사과와 노란·초록 사과를 섞어 찍어야 경계가 의미 있다")
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
    return th


# ---------------------------------------------------------------- 4. truth

def truth_root(root: Path, calib: CardCalib | None, p: InspectParams) -> tuple[list[dict], dict]:
    """root/<사과>/snap_*.npz — 첫 장은 자연스럽게 놓인 자세(탑뷰 판정에 쓰는 장), 나머지는 굴려서 다른 면이 위로."""
    rows = []
    for d in sorted(x for x in root.iterdir() if x.is_dir() and not x.name.startswith("_")):
        views = []
        for path, raw, depth, intr in load_snaps(d):
            color = apply_gain(raw, calib)
            dets = segment_apples(color, depth, intr)
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
    if not rows:
        raise SystemExit("처리한 사과가 없다")
    diffs = np.abs([r["diff"] for r in rows])
    p90 = float(np.percentile(diffs, 90))
    summary = {"apples": len(rows), "mae": float(diffs.mean()), "p90": p90,
               "suggested_top_view_error": max(0.03, math.ceil(p90 * 100) / 100)}
    return rows, summary


def cmd_truth(root: Path, card: str | None, variety: str | None) -> dict:
    p = params(variety)
    rows, s = truth_root(root, find_card(root, card), p)
    write_csv(analysis_dir(root) / "truth.csv", rows)
    print(f"{'사과':<12}{'뷰':>3}{'탑뷰':>8}{'전표면':>8}{'차이':>8}")
    for r in rows:
        print(f"{r['apple']:<12}{r['views']:>3}{r['top_ratio'] * 100:>7.0f}%{r['full_ratio'] * 100:>7.0f}%{r['diff'] * 100:>+7.0f}%p")
    print(f"\n사과 {s['apples']}개 — 탑뷰 오차 평균 {s['mae'] * 100:.1f}%p, 90% 지점 {s['p90'] * 100:.1f}%p")
    print(f"제안: configs/varieties.yaml 의 top_view_error: {s['suggested_top_view_error']:.2f}  (현재 {p.top_view_error:.2f})")
    print(f"CSV: {analysis_dir(root) / 'truth.csv'}")
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
        print(f"\n육안 상/중 {n}개 기준 — 전표면 일치 {stats['full_agree'] / n:.0%}, "
              f"탑뷰 일치 {stats['top_agree'] / n:.0%} (리젝트로 보류 {stats['top_reject'] / n:.0%})")
    return stats


def _cr(ratio: float, p: InspectParams):
    from aple.schema import ColorRatio, InspectMode
    return ColorRatio(0, 0, InspectMode.TOP_SINGLE, ratio, max(0.0, ratio - p.top_view_error),
                      min(1.0, ratio + p.top_view_error), 0.25, 0.0, 1, p.variety)


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("card")
    a.add_argument("folder", type=Path)
    a.add_argument("--roi", help="x,y,w,h (화면 없이 지정할 때)")
    a.add_argument("--file", help="기준으로 쓸 스냅샷 (기본: 폴더 첫 장)")
    for name in ("analyze", "suggest", "truth"):
        b = sub.add_parser(name)
        b.add_argument("folder", type=Path)
        b.add_argument("--card", help="card.json 경로 (기본: 폴더 안 card.json)")
        b.add_argument("--variety", choices=["fuji", "hongro"])
    c = sub.add_parser("compare")
    c.add_argument("truth_csv", type=Path)
    c.add_argument("--labels", type=Path, required=True)
    c.add_argument("--variety", choices=["fuji", "hongro"])
    args = ap.parse_args(argv)

    if args.cmd == "card":
        cmd_card(args.folder, args.roi, args.file)
    elif args.cmd == "analyze":
        cmd_analyze(args.folder, args.card, args.variety)
    elif args.cmd == "suggest":
        cmd_suggest(args.folder, args.card, args.variety)
    elif args.cmd == "truth":
        cmd_truth(args.folder, args.card, args.variety)
    else:
        cmd_compare(args.truth_csv, args.labels, args.variety)


if __name__ == "__main__":
    main()
