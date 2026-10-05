"""사과 몇 개로 YOLO 데이터셋 만들기 — 자동 라벨·증강·출력·Piper Studio 업로드 검증."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from tests.synthetic import INTR, Apple, half_red, render, side_band
from tools import make_yolo_dataset as my


def save(folder, i, color, depth):
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"snap_{i:04d}.npz", color=color, depth_m=depth,
                        intrinsics=np.array(INTR), stamp_s=float(i))


def scene(xs, seed=0, y=0.0):
    return render([Apple((x, y, 0.60), 0.038, side_band(0.2 * (i - 1))) for i, x in enumerate(xs)], seed=seed)


@pytest.fixture
def sessions(tmp_path):
    """세션 2개(사과 배치 여러 장) + 빈 배경 세션."""
    layouts = [(-0.12, 0.0, 0.12), (-0.06, 0.08), (0.10,), (-0.15, -0.03, 0.09)]
    for s in ("s1", "s2"):
        for i, xs in enumerate(layouts):
            c, d, _ = scene(xs, seed=i, y=0.03 if s == "s2" else 0.0)
            save(tmp_path / s, i, c, d)
    for i in range(2):
        c, d, _ = render([], seed=i)
        save(tmp_path / "empty", i, c, d)
    return tmp_path


def test_autolabel_matches_true_apple_regions(tmp_path):
    color, depth, masks = scene((-0.12, 0.0, 0.12))
    save(tmp_path / "s", 0, color, depth)
    samples, empties, dropped = my.autolabel_folder(tmp_path / "s")
    assert len(samples) == 1 and dropped == 0 and not empties
    got = samples[0].masks
    assert len(got) == 3
    for g, m in zip(got, masks):
        assert (g & m).sum() / (g | m).sum() > 0.85


def test_touching_apples_scene_is_dropped(tmp_path):
    color, depth, _ = render([Apple((-0.037, 0.0, 0.60), 0.038, half_red), Apple((0.037, 0.0, 0.60), 0.038, half_red)])
    save(tmp_path / "s", 0, color, depth)
    samples, _, dropped = my.autolabel_folder(tmp_path / "s")
    assert not samples and dropped == 1                     # 붙은 사과는 라벨이 틀리므로 쓰지 않는다


def test_label_lines_are_normalized_yolo():
    m = np.zeros((100, 200), bool)
    m[20:60, 50:90] = True
    assert my.bbox_line(m) == "0 0.350000 0.400000 0.200000 0.400000"
    poly = my.poly_line(m).split()
    assert poly[0] == "0" and len(poly) >= 7 and all(0 <= float(v) <= 1 for v in poly[1:])


def test_copy_paste_creates_labeled_scenes(sessions):
    real, _, _ = my.autolabel_folder(sessions / "s1")
    _, empties, _ = my.autolabel_folder(sessions / "empty")
    synth = my.synthesize(empties, my.cutouts_from(real), n=20, seed=3)
    assert len(synth) >= 18
    for s in synth:
        assert 1 <= len(s.masks) <= 5 and all(m.sum() > 100 for m in s.masks)
        for a in range(len(s.masks)):                       # 서로 덮인 부분은 한쪽에만 남는다
            for b in range(a + 1, len(s.masks)):
                assert not (s.masks[a] & s.masks[b]).any()
        # 붙인 자리는 배경과 다르다 (사과 픽셀이 실제로 들어갔다)
        assert np.abs(s.image[s.masks[0]].astype(int) - 150).mean() > 20


def test_main_writes_ultralytics_dataset_with_session_split(sessions, tmp_path):
    out = tmp_path / "ds"
    r = my.main([str(sessions / "s1"), str(sessions / "s2"), "--out", str(out), "--synth", "10",
                 "--backgrounds", str(sessions / "empty")])
    assert r["val"] == 4 and r["synth"] >= 9 and r["train"] == 4 + r["synth"]
    assert all(p.name.startswith("s2__") for p in (out / "images" / "val").iterdir())   # 검증 = 실제 세션만
    assert len(list((out / "labels" / "train").glob("*.txt"))) == r["train"]
    assert "names:\n  0: apple" in (out / "data.yaml").read_text(encoding="utf-8")


class FakePiper(BaseHTTPRequestHandler):
    """Piper Studio YOLO API 흉내 — 요청 형식만 확인한다."""
    store = {"datasets": [], "images": [], "labels": {}}

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def do_GET(self):
        self._json({"datasets": [{"name": n} for n in self.store["datasets"]]})

    def do_POST(self):
        body = self._body()
        if self.path == "/api/yolo/datasets":
            self.store["datasets"].append(json.loads(body)["name"])
            return self._json({"ok": True})
        assert self.path.startswith("/api/yolo/datasets/apple_v1/images?type=upload")
        assert self.headers["Content-Type"] == "image/jpeg" and body[:2] == b"\xff\xd8"   # JPEG
        name = f"img{len(self.store['images'])}.jpg"
        self.store["images"].append(name)
        self._json({"file": name})

    def do_PUT(self):
        name = self.path.rsplit("/", 1)[1]
        self.store["labels"][name] = json.loads(self._body())["boxes"]
        self._json({"boxes": self.store["labels"][name]})

    def log_message(self, *a):
        pass


def test_upload_to_piper_studio_api(sessions):
    srv = HTTPServer(("127.0.0.1", 0), FakePiper)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        real, _, _ = my.autolabel_folder(sessions / "s1")
        n = my.upload_piper(f"http://127.0.0.1:{srv.server_port}", "apple_v1", real)
    finally:
        srv.shutdown()
    st = FakePiper.store
    assert n == len(real) and st["datasets"] == ["apple_v1"]
    assert len(st["labels"]) == n
    first = st["labels"]["img0.jpg"]
    assert len(first) == 3 and set(first[0]) == {"cls", "cx", "cy", "w", "h"}
