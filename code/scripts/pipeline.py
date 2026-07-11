"""
pipeline.py — the whole tool. Video in -> highlight clips out.

One detection pass over the video (simple_make: enlarged rim zone + a
sustained ABOVE->BELOW pass to confirm the ball went through), then:
  * de-dup makes that fire within --min-gap seconds (miscount guard), and
  * cut a [make - pre, make + post] clip around each surviving make.

Because the clip reaches ~10s back by default, even when the de-dup keeps the
"wrong" one of a close pair, the real make still lands inside the clip.

Usage:
    python scripts/pipeline.py --source ~/Desktop/TestClips/broadcast_lal_vs_gs_2makes.mp4
    python scripts/pipeline.py --source game.mp4 --pre 10 --post 3 --out output/clips
"""
import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from simple_make import detect, DEFAULT_WEIGHTS
from make_detector import dedup_makes
from clip_extractor import extract_all

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description="Made Shot Highlighter: video in -> clips out")
    ap.add_argument("--source", required=True, help="input game video")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "output", "clips"))
    ap.add_argument("--conf", type=float, default=0.12, help="detection confidence threshold")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--pre", type=float, default=10.0, help="seconds before the make")
    ap.add_argument("--post", type=float, default=3.0, help="seconds after the make")
    ap.add_argument("--min-gap", type=float, default=7.0, dest="min_gap",
                    help="cancel the earlier of two makes closer than this many seconds "
                         "(miscount guard; 0 disables)")
    # make-logic knobs (forwarded to simple_make.detect)
    ap.add_argument("--cooldown", type=float, default=1.5, dest="cooldown_s")
    ap.add_argument("--min-above", type=int, default=3, dest="min_above",
                    help="min frames the ball must be seen ABOVE the rim before a make")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="inference resolution (default: auto — 1280 for >=2000px sources, else 640)")
    ap.add_argument("--no-frames", action="store_true",
                    help="skip saving an annotated thumbnail per make")
    ap.add_argument("--rim-source", default="rfdetr", choices=["local", "rfdetr"], dest="rim_source",
                    help="'local' uses the same YOLO weights for ball+rim; 'rfdetr' keeps ball "
                         "detection local but fetches rim boxes from the local RF-DETR checkpoint")
    ap.add_argument("--rim-weights", default=None, dest="rim_weights",
                    help="path to the RF-DETR checkpoint when --rim-source=rfdetr")
    ap.add_argument("--rim-stride", type=int, default=6, dest="rim_stride",
                    help="run the RF-DETR rim model every N frames (holds position in between)")
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        raise SystemExit(f"Weights not found: {args.weights}\nTrain first (scripts/train.py).")

    frames_dir = None if args.no_frames else os.path.join(args.out, "frames")

    print(f"Analyzing {args.source} ...")
    detect_kwargs = dict(
        weights=args.weights, conf=args.conf, device=args.device,
        out_dir=frames_dir, cooldown_s=args.cooldown_s, min_above=args.min_above,
        imgsz=args.imgsz, verbose=True,
        rim_source=args.rim_source, rim_stride=args.rim_stride,
    )
    if args.rim_weights:
        detect_kwargs["rim_weights"] = args.rim_weights
    makes, fps = detect(args.source, **detect_kwargs)

    if not makes:
        print("No makes detected.")
        return

    # Miscount guard: drop the earlier of any two makes within --min-gap seconds.
    if args.min_gap > 0:
        deduped = dedup_makes(makes, min_gap_s=args.min_gap)
        for m in makes:
            if m not in deduped:
                print(f"  (miscount guard: canceled make at t={m['time']:.2f}s — "
                      f"another make within {args.min_gap:.0f}s)")
                # Keep the canceled make's thumbnail (renamed) so the full trace
                # stays visible for debugging — don't silently delete it.
                img = m.get("image")
                if img and os.path.exists(img):
                    d, b = os.path.split(img)
                    canceled = os.path.join(d, "CANCELED_" + b)
                    os.replace(img, canceled)
        makes = deduped

    print(f"\nCutting {len(makes)} clip(s) into {args.out} ...")
    paths = extract_all(args.source, makes, args.out, pre=args.pre, post=args.post)
    for m, p in zip(makes, paths):
        thumb = f"  [thumb: {os.path.basename(m['image'])}]" if m.get("image") else ""
        print(f"  t={m['time']:6.2f}s  ->  {os.path.basename(p)}{thumb}")
    print(f"\nDone. {len(paths)} highlight clip(s) in {args.out}")


if __name__ == "__main__":
    main()
