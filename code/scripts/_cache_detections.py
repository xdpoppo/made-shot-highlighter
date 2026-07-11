"""
_cache_detections.py — run ball+rim inference ONCE per clip and pickle the
per-frame raw boxes, so make-logic parameters can be swept offline without
re-running the (slow) detectors each time. Debug/tuning helper, not shipped.
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
from collections import deque
from make_detector import _pick_centers
from simple_make import _load_rfdetr_rim_model, _rfdetr_rim_box, DEFAULT_WEIGHTS
from ultralytics import YOLO

CACHE_DIR = "/tmp/det_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def cache_clip(path, rim_stride=6):
    name = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(CACHE_DIR, name + ".pkl")
    if os.path.exists(out):
        print(f"  [skip] {name} already cached")
        return
    model = YOLO(DEFAULT_WEIGHTS)
    names = model.names
    rf_model = _load_rfdetr_rim_model()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    imgsz = 1280 if W >= 2000 else 640
    ball_hist = deque(maxlen=90)
    ball_raw, rim_raw = [], []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        r = model.predict(source=frame, conf=0.12, device="mps", verbose=False, imgsz=imgsz)[0]
        if len(ball_hist) >= 2:
            (lf, x1, y1), (_, x0, y0) = ball_hist[-1], ball_hist[-2]
            pred = (x1 + (x1 - x0), y1 + (y1 - y0))
        elif ball_hist:
            lf, x1, y1 = ball_hist[-1]
            pred = (x1, y1)
        else:
            lf, pred = idx, None
        gap = max(1, idx - lf)
        dyn = min(600, 100 + 60 * gap)
        ball_box, _ = _pick_centers(r.boxes, names, prev_ball=pred, max_jump=dyn, ball_hist=ball_hist)
        rim_box = _rfdetr_rim_box(rf_model, frame, 0.2) if idx % rim_stride == 0 else None
        if ball_box is not None:
            cx = (ball_box[0] + ball_box[2]) / 2
            cy = (ball_box[1] + ball_box[3]) / 2
            ball_hist.append((idx, cx, cy))
        ball_raw.append(list(ball_box) if ball_box is not None else None)
        rim_raw.append(list(rim_box) if rim_box is not None else None)
        idx += 1
    cap.release()
    with open(out, "wb") as f:
        pickle.dump({"ball_raw": ball_raw, "rim_raw": rim_raw, "fps": fps, "W": W}, f)
    print(f"  [done] {name}: {idx} frames -> {out}")


if __name__ == "__main__":
    clips = sys.argv[1:]
    for c in clips:
        print(f"Caching {c} ...")
        cache_clip(c)
    print("All done.")
