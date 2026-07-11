"""
test_roboflow_rim.py — sanity-check the Roboflow-hosted rim detector (v3, RF-DETR)
against local test clips.

Samples frames from a video at a fixed stride, runs each through Roboflow's
hosted inference API, draws the returned rim boxes, and saves annotated
frames + a summary of detection rate / confidence.

Usage:
    python scripts/test_roboflow_rim.py --source data/videos/TestClip.mp4
"""
import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--version", type=int, default=3)
    ap.add_argument("--stride", type=int, default=30, help="sample every Nth frame")
    ap.add_argument("--conf", type=int, default=40, help="confidence threshold (0-100)")
    ap.add_argument("--out", default=None, help="output dir for annotated samples")
    args = ap.parse_args()

    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    api_key = os.environ["ROBOFLOW_API_KEY"]
    workspace = os.environ["ROBOFLOW_WORKSPACE"]
    project_name = os.environ["ROBOFLOW_PROJECT"]

    clip_name = os.path.splitext(os.path.basename(args.source))[0]
    out_dir = args.out or os.path.join(PROJECT_ROOT, "output", "roboflow_test", clip_name)
    os.makedirs(out_dir, exist_ok=True)

    import roboflow
    rf = roboflow.Roboflow(api_key=api_key)
    version = rf.workspace(workspace).project(project_name).version(args.version)
    model = version.model

    cap = cv2.VideoCapture(args.source)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    frame_idx = 0
    sampled = 0
    detected = 0
    confidences = []

    tmp_frame_path = os.path.join(out_dir, "_tmp_frame.jpg")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.stride == 0:
            cv2.imwrite(tmp_frame_path, frame)
            try:
                preds = model.predict(tmp_frame_path, confidence=args.conf, overlap=30).json()["predictions"]
            except Exception as e:
                print(f"  frame {frame_idx}: inference error -> {e}")
                preds = []

            rim_preds = [p for p in preds if p["class"] == "rim"]
            sampled += 1
            if rim_preds:
                detected += 1

            colors = {"rim": (0, 0, 255), "ball": (0, 140, 255)}
            for p in preds:
                if p["class"] == "rim":
                    confidences.append(p["confidence"])
                color = colors.get(p["class"], (255, 255, 255))
                x1 = int(p["x"] - p["width"] / 2)
                y1 = int(p["y"] - p["height"] / 2)
                x2 = int(p["x"] + p["width"] / 2)
                y2 = int(p["y"] + p["height"] / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{p['class']} {p['confidence']:.2f}", (x1, max(14, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            out_path = os.path.join(out_dir, f"frame_{frame_idx:05d}_t{frame_idx/fps:.2f}s.jpg")
            cv2.imwrite(out_path, frame)
            print(f"  frame {frame_idx:5d} (t={frame_idx/fps:5.2f}s): "
                  f"{len(rim_preds)} rim box(es)" + (f" conf={rim_preds[0]['confidence']:.2f}" if rim_preds else "")
                  + f" | {len(preds) - len(rim_preds)} other")

        frame_idx += 1

    cap.release()
    if os.path.exists(tmp_frame_path):
        os.remove(tmp_frame_path)

    print("\n" + "=" * 50)
    print(f"Clip: {args.source}")
    print(f"Sampled frames: {sampled} (stride {args.stride}, total {total})")
    print(f"Frames with rim detected: {detected} ({100*detected/max(1,sampled):.1f}%)")
    if confidences:
        print(f"Avg confidence: {sum(confidences)/len(confidences):.3f}")
    print(f"Annotated samples: {out_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
