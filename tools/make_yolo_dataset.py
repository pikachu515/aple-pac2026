"""사과 몇 개로 YOLO 검출 데이터셋 만들기 — 깊이 자동 라벨 + 붙여넣기 증강 + Piper Studio 업로드.

사과가 적어도(5~10개) 학습 데이터를 만들 수 있게 하는 세 가지:

1. **깊이 자동 라벨** — D435 스냅샷(.npz: 컬러 + 정렬 깊이)에서 테이블보다 솟은 덩어리를 사과로 보고
   경계상자·윤곽을 만든다 (aple/inspection/standalone.py). 손으로 박스를 그리지 않는다.
   사과끼리 붙어 한 덩어리가 된 장면은 크기로 걸러 버린다.
2. **붙여넣기 증강 (copy-paste)** — 실제로 찍은 사과를 오려 빈 테이블·트레이·상자 사진에 무작위
   위치·회전·크기·밝기로 붙인다. 라벨은 붙인 위치에서 바로 나온다. 실제 사과 픽셀이므로 색·질감이 진짜다.
   ⚠ 그림자·가림은 부정확하므로 학습에만 섞고, 검증은 실제 사진으로만 한다.
3. **Piper Studio 업로드** — 이미지와 라벨을 Piper Studio YOLO 데이터셋에 REST 로 넣는다
   (POST /api/yolo/datasets/{이름}/images, PUT .../labels/{파일}). 이후 Piper Studio 화면에서 학습하면
   결과 가중치가 모델 목록에 자동 등록된다.

    # ① 자동 라벨 + 증강 → ultralytics 형식 (Colab·로컬 학습용)
    python tools/make_yolo_dataset.py data/snapshots/1006_* --out data/yolo/apple_v1 --synth 300 \\
        --backgrounds data/snapshots/1006_empty

    # ② 같은 내용을 Piper Studio 데이터셋으로 업로드
    python tools/make_yolo_dataset.py data/snapshots/1006_* --synth 300 --backgrounds data/snapshots/1006_empty \\
        --upload http://localhost --dataset apple_v1

폴더 하나 = 촬영 세션 하나. 검증 세트는 세션 단위로 떼어 낸다 (같은 세션 연속 장면이 양쪽에 들어가면 점수가 부풀려진다).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aple.inspection.standalone import segment_apples  # noqa: E402

CLASSES = ["apple"]


@dataclass
class Sample:
    image: np.ndarray                  # BGR
    masks: list[np.ndarray]            # 사과마다 bool 마스크
    group: str                         # 세션(폴더) 이름 — 검증 분할 단위
    name: str


# ---------------------------------------------------------------- 1. 자동 라벨

def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, tuple]:
    with np.load(path) as z:
        return z["color"], z["depth_m"], tuple(float(v) for v in z["intrinsics"])


def roundness(mask: np.ndarray) -> float:
    """면적 / 외접원 넓이. 사과 하나는 0.85 안팎, 두 개가 붙으면 0.5 안팎."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (_, _), r = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
    return float(mask.sum() / (np.pi * r * r)) if r > 0 else 0.0


def autolabel_folder(folder: Path, max_diam_m: float = 0.11,
                     min_roundness: float = 0.7) -> tuple[list[Sample], list[np.ndarray], int]:
    """→ (사과가 있는 장면, 사과가 없는 장면(배경 후보), 버린 장면 수)."""
    samples, empties, dropped = [], [], 0
    for p in sorted(folder.glob("*.npz")):
        color, depth, intr = load_npz(p)
        try:
            dets = segment_apples(color, depth, intr, diam_range_m=(0.03, 0.15))
        except ValueError:
            dropped += 1
            continue
        f = (intr[0] + intr[1]) / 2
        big = [d for d in dets if 2 * np.sqrt(d.mask.sum() / np.pi) * float(np.median(depth[d.mask & (depth > 0)])) / f > max_diam_m]
        merged = [d for d in dets if roundness(d.mask) < min_roundness]
        if big or merged:                             # 붙은 사과 → 라벨이 틀리므로 장면을 통째로 버린다
            dropped += 1
            continue
        if dets:
            samples.append(Sample(color, [d.mask for d in dets], folder.name, p.stem))
        else:
            empties.append(color)
    return samples, empties, dropped


def bbox_line(mask: np.ndarray) -> str:
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}"


def poly_line(mask: np.ndarray, eps_px: float = 1.5) -> str | None:
    H, W = mask.shape
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), eps_px, True).reshape(-1, 2)
    if len(c) < 3:
        return None
    return "0 " + " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in c)


# ---------------------------------------------------------------- 2. 붙여넣기 증강

@dataclass
class Cutout:
    rgb: np.ndarray       # 사과를 감싼 사각형 (BGR)
    alpha: np.ndarray     # 같은 크기 float 0~1


def cutouts_from(samples: list[Sample]) -> list[Cutout]:
    out = []
    for s in samples:
        for m in s.masks:
            ys, xs = np.nonzero(m)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            a = cv2.GaussianBlur(m[y0:y1, x0:x1].astype(np.float32), (3, 3), 0)   # 가장자리 1픽셀 부드럽게
            out.append(Cutout(s.image[y0:y1, x0:x1].copy(), a))
    return out


def _transform(c: Cutout, rng, scale_jitter=0.10, bright_jitter=0.10) -> Cutout:
    s = 1 + rng.uniform(-scale_jitter, scale_jitter)
    ang = rng.uniform(0, 360)
    h, w = c.alpha.shape
    side = int(np.ceil(np.hypot(h, w) * s)) + 2
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, s)
    M[:, 2] += [(side - w) / 2, (side - h) / 2]
    rgb = cv2.warpAffine(c.rgb, M, (side, side), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    alpha = cv2.warpAffine(c.alpha, M, (side, side), flags=cv2.INTER_LINEAR)
    gain = 1 + rng.uniform(-bright_jitter, bright_jitter)               # 색상각은 그대로, 밝기만
    rgb = np.clip(rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return Cutout(rgb, alpha)


def synthesize(backgrounds: list[np.ndarray], cuts: list[Cutout], n: int, seed: int = 0,
               max_apples: int = 5, max_overlap: float = 0.15) -> list[Sample]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        bg = backgrounds[rng.integers(len(backgrounds))].copy()
        H, W = bg.shape[:2]
        canvas = bg.astype(np.float32)
        taken = np.zeros((H, W), bool)
        masks = []
        for _ in range(int(rng.integers(1, max_apples + 1))):
            c = _transform(cuts[rng.integers(len(cuts))], rng)
            h, w = c.alpha.shape
            for _try in range(20):                                       # 너무 겹치지 않는 자리
                y, x = int(rng.integers(0, H - h)), int(rng.integers(0, W - w))
                m = np.zeros((H, W), bool)
                m[y:y + h, x:x + w] = c.alpha > 0.5
                if (m & taken).sum() <= max_overlap * m.sum():
                    break
            else:
                continue
            a = c.alpha[..., None]
            canvas[y:y + h, x:x + w] = canvas[y:y + h, x:x + w] * (1 - a) + c.rgb.astype(np.float32) * a
            for prev in masks:                                           # 위에 덮인 부분은 앞 사과에서 뺀다
                prev &= ~m
            masks.append(m)
            taken |= m
        masks = [m for m in masks if m.sum() > 100]
        if masks:
            out.append(Sample(np.clip(canvas, 0, 255).astype(np.uint8), masks, "synth", f"synth_{i:05d}"))
    return out


# ---------------------------------------------------------------- 출력

def split(samples: list[Sample], val_groups: set[str]) -> tuple[list[Sample], list[Sample]]:
    return ([s for s in samples if s.group not in val_groups],
            [s for s in samples if s.group in val_groups])


def write_ultralytics(out: Path, train: list[Sample], val: list[Sample], task: str = "detect") -> Path:
    for part, items in (("train", train), ("val", val)):
        (out / "images" / part).mkdir(parents=True, exist_ok=True)
        (out / "labels" / part).mkdir(parents=True, exist_ok=True)
        for s in items:
            stem = f"{s.group}__{s.name}"
            cv2.imwrite(str(out / "images" / part / f"{stem}.jpg"), s.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            lines = [bbox_line(m) if task == "detect" else poly_line(m) for m in s.masks]
            (out / "labels" / part / f"{stem}.txt").write_text("\n".join(x for x in lines if x), encoding="utf-8")
    yaml_path = out / "data.yaml"
    yaml_path.write_text(f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
                         f"names:\n  0: apple\n", encoding="utf-8")
    return yaml_path


def upload_piper(base_url: str, dataset: str, samples: list[Sample], timeout: float = 30.0) -> int:
    """Piper Studio YOLO 데이터셋에 이미지·라벨을 넣는다. 데이터셋이 없으면 만든다."""
    api = base_url.rstrip("/") + "/api/yolo/datasets"

    def call(method, url, body=None, ctype="application/json"):
        req = urllib.request.Request(url, data=body, method=method, headers={"Content-Type": ctype})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")

    names = [d.get("name") for d in call("GET", api).get("datasets", [])]
    if dataset not in names:
        call("POST", api, json.dumps({"name": dataset, "classes": CLASSES}).encode())
    q = urllib.parse.quote(dataset)
    n = 0
    for s in samples:
        ok, jpg = cv2.imencode(".jpg", s.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        fname = call("POST", f"{api}/{q}/images?type=upload", jpg.tobytes(), "image/jpeg")["file"]
        boxes = []
        for line in (bbox_line(m) for m in s.masks):
            _, cx, cy, w, h = line.split()
            boxes.append({"cls": 0, "cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h)})
        call("PUT", f"{api}/{q}/labels/{urllib.parse.quote(fname)}", json.dumps({"boxes": boxes}).encode())
        n += 1
    return n


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", type=Path, help="스냅샷 폴더 (폴더 = 촬영 세션)")
    ap.add_argument("--out", type=Path, help="ultralytics 형식 출력 폴더")
    ap.add_argument("--task", choices=["detect", "segment"], default="detect")
    ap.add_argument("--val", nargs="*", default=None, help="검증 세트로 뗄 세션 폴더 이름 (기본: 마지막 폴더)")
    ap.add_argument("--synth", type=int, default=0, help="붙여넣기 증강 장수 (학습 세트에만)")
    ap.add_argument("--backgrounds", nargs="*", type=Path, default=[], help="빈 테이블·트레이·상자 스냅샷 폴더")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--upload", help="Piper Studio 주소 (예: http://localhost)")
    ap.add_argument("--dataset", default="apple_v1", help="Piper Studio 데이터셋 이름")
    args = ap.parse_args(argv)

    real, empties, dropped = [], [], 0
    for f in args.folders:
        s, e, d = autolabel_folder(f)
        real += s
        empties += e
        dropped += d
        print(f"{f.name}: 사과 장면 {len(s)}장 (사과 {sum(len(x.masks) for x in s)}개), 빈 장면 {len(e)}장, 버림 {d}장")
    for f in args.backgrounds:
        _, e, _ = autolabel_folder(f)
        empties += e
    if not real:
        raise SystemExit("자동 라벨된 장면이 없다 — 깊이(USB 3)와 사과 간격을 확인")

    groups = sorted({s.group for s in real})
    val_groups = set(args.val) if args.val is not None else ({groups[-1]} if len(groups) > 1 else set())
    train, val = split(real, val_groups)
    synth = []
    if args.synth:
        if not empties:
            raise SystemExit("붙여넣기 증강에는 빈 배경 장면이 필요하다 — 사과 없이 몇 장 찍어 --backgrounds 로 넘긴다")
        synth = synthesize(empties, cutouts_from(train), args.synth, args.seed)
        train = train + synth
    print(f"학습 {len(train)}장 (실제 {len(train) - len(synth)} + 증강 {len(synth)}), 검증 {len(val)}장 (실제만, 세션 {sorted(val_groups)})")
    if not val:
        print("  [주의] 검증 세트가 비었다 — 촬영 세션(폴더)을 2개 이상으로 나눠 찍는다")

    result = {"train": len(train), "val": len(val), "synth": len(synth), "dropped": dropped}
    if args.out:
        result["data_yaml"] = str(write_ultralytics(args.out, train, val, args.task))
        print(f"ultralytics 데이터셋: {result['data_yaml']}")
    if args.upload:
        result["uploaded"] = upload_piper(args.upload, args.dataset, train + val)
        print(f"Piper Studio '{args.dataset}' 에 {result['uploaded']}장 업로드 — 화면에서 확인 후 학습")
    return result


if __name__ == "__main__":
    main()
