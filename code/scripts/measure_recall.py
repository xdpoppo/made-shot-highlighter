"""
measure_recall.py — true recall of ball/rim detection, not just hit-rate.

detect_video.py reports `detected / all frames`, which conflates two things:
frames where the object genuinely isn't in view (camera framing) and frames
where it IS in view but the model missed it. This script measures the second
one directly: `detected / frames-where-object-is-actually-visible`, using a
hand-labeled ground-truth visibility file.

Usage:
    # 1. extract a sample of frames to label by eye
    python scripts/measure_recall.py --source data/videos/clip.mp4 --extract-out /tmp/sample_frames --stride 20

    # 2. after writing data/labels/testclip_visibility.json by hand, measure recall
    python scripts/measure_recall.py --source data/videos/clip.mp4 --labels data/labels/testclip_visibility.json
"""
import argparse
import json
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "models", "ball_detector_v2025", "weights", "best.pt")


def extract_sample_frames(source, out_dir, stride):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(source)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    saved = []
    for idx in range(0, total, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        path = os.path.join(out_dir, f"f{idx:05d}.jpg")
        cv2.imwrite(path, frame)
        saved.append(idx)
    cap.release()
    print(f"Extracted {len(saved)} frames (every {stride}th of {total}) to {out_dir}")
    print("Label each as {\"ball_visible\": bool, \"rim_visible\": bool} keyed by frame index.")


def measure(source, labels_path, weights, conf, device):
    with open(labels_path) as f:
        labels = json.load(f)

    model = YOLO(weights)
    names = model.names

    cap = cv2.VideoCapture(source)
    stats = {
        "ball": {"visible": 0, "visible_detected": 0, "hidden": 0, "hidden_detected": 0},
        "rim": {"visible": 0, "visible_detected": 0, "hidden": 0, "hidden_detected": 0},
    }

    for frame_idx_str, gt in sorted(labels.items(), key=lambda kv: int(kv[0])):
        frame_idx = int(frame_idx_str)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        results = model.predict(source=frame, conf=conf, device=device, verbose=False)
        found = {names[int(b.cls)].lower() for b in results[0].boxes}

        for cls_names, key in ((("ball",), "ball_visible"), (("rim", "hoop"), "rim_visible")):
            visible = bool(gt.get(key, False))
            detected = any(c in found for c in cls_names)
            bucket = stats["ball" if "ball" in cls_names else "rim"]
            if visible:
                bucket["visible"] += 1
                bucket["visible_detected"] += int(detected)
            else:
                bucket["hidden"] += 1
                bucket["hidden_detected"] += int(detected)

    cap.release()

    print("\n" + "=" * 60)
    print(f"Labeled frames: {len(labels)}")
    for cls in ("ball", "rim"):
        s = stats[cls]
        recall = 100 * s["visible_detected"] / s["visible"] if s["visible"] else float("nan")
        fpr = 100 * s["hidden_detected"] / s["hidden"] if s["hidden"] else float("nan")
        print(f"\n{cls}:")
        print(f"  visible in {s['visible']} frames -> detected in {s['visible_detected']} "
              f"(true recall: {recall:.1f}%)")
        print(f"  not visible in {s['hidden']} frames -> detected in {s['hidden_detected']} "
              f"(false positive rate: {fpr:.1f}%)")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--stride", type=int, default=20, help="frame sampling stride for extraction")
    ap.add_argument("--extract-out", default=None, help="if set, only extract sample frames here")
    ap.add_argument("--labels", default=None, help="path to ground-truth visibility JSON")
    args = ap.parse_args()

    if args.extract_out:
        extract_sample_frames(args.source, args.extract_out, args.stride)
        return

    if not args.labels:
        raise SystemExit("Provide --extract-out to sample frames, or --labels to measure recall.")
    measure(args.source, args.labels, args.weights, args.conf, args.device)


if __name__ == "__main__":
    main()
