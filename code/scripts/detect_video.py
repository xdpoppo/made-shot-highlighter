"""
detect_video.py — Phase 4: run the trained detector over a video and visualize.

Draws ball/human/rim boxes on every frame, writes an annotated preview video,
saves a few sample frames, and reports how reliably the ball and rim are found
(so we know the detector is solid before trusting the make/miss logic).

Usage:
    python scripts/detect_video.py --source data/videos/clip.mp4
    python scripts/detect_video.py --source data/videos/clip.mp4 --weights models/ball_rim_v1/weights/best.pt --conf 0.25
"""
import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "models", "ball_detector_v2025", "weights", "best.pt")
COLORS = {"ball": (0, 140, 255), "human": (0, 200, 0), "rim": (0, 0, 255)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="input video path")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "output", "detect"))
    ap.add_argument("--vid-stride", type=int, default=1, dest="vid_stride",
                    help="process every Nth frame (N=3 ~3x faster)")
    args = ap.parse_args()
    vid_stride = max(1, args.vid_stride)

    os.makedirs(args.out, exist_ok=True)
    model = YOLO(args.weights)
    names = model.names  # {0:'ball',...}

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_video = os.path.join(args.out, "annotated.mp4")
    # fps/vid_stride keeps playback speed correct since we only write kept frames.
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps / vid_stride, (W, H))

    frame_idx = ball_frames = rim_frames = 0
    sample_every = max(1, (total // vid_stride) // 8) if total else 60

    for r in model.predict(source=args.source, stream=True, conf=args.conf,
                           device=args.device, vid_stride=vid_stride, verbose=False):
        frame = r.orig_img.copy()
        found = set()
        for b in r.boxes:
            cls = names[int(b.cls)]
            found.add(cls.lower())
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            color = COLORS.get(cls.lower(), (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls} {float(b.conf):.2f}", (x1, max(14, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if "ball" in found:
            ball_frames += 1
        if "rim" in found or "hoop" in found:
            rim_frames += 1
        writer.write(frame)
        if frame_idx % sample_every == 0:
            cv2.imwrite(os.path.join(args.out, f"frame_{frame_idx:05d}.jpg"), frame)
        frame_idx += 1

    cap.release()
    writer.release()

    n = max(1, frame_idx)
    print("\n" + "=" * 50)
    print(f"Frames processed: {frame_idx} (stride {vid_stride})")
    print(f"Ball detected in: {ball_frames} ({100*ball_frames/n:.1f}% of frames)")
    print(f"Rim  detected in: {rim_frames} ({100*rim_frames/n:.1f}% of frames)")
    print(f"Annotated video : {out_video}")
    print(f"Sample frames   : {args.out}")
    print("=" * 50)


if __name__ == "__main__":
    main()
