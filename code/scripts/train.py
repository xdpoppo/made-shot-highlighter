"""
train.py — Phase 3: fine-tune YOLO11 on the basketball ball/human/rim dataset.

Runs locally on Apple Silicon via the MPS backend. Uses transfer learning from
the pretrained yolo11n.pt weights (not from scratch).

Usage:
    python scripts/train.py --smoke              # 1-epoch sanity check
    python scripts/train.py                       # full run (default 50 epochs)
    python scripts/train.py --epochs 25 --batch 16
"""
import argparse
import os

# Let unsupported MPS ops fall back to CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import yaml
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_local_yaml(dataset_dir):
    """Rewrite data.yaml with absolute split paths (Roboflow's ../ paths are fragile)."""
    DATASET_DIR = dataset_dir
    with open(os.path.join(DATASET_DIR, "data.yaml")) as f:
        orig = yaml.safe_load(f)
    local = {
        "path": DATASET_DIR,
        "train": os.path.join(DATASET_DIR, "train", "images"),
        "val": os.path.join(DATASET_DIR, "valid", "images"),
        "test": os.path.join(DATASET_DIR, "test", "images"),
        "nc": orig["nc"],
        "names": orig["names"],
    }
    out = os.path.join(DATASET_DIR, "data_local.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(local, f, sort_keys=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache", default=None,
                    help="image cache: 'ram', 'disk', or omit for none (speeds data loading)")
    ap.add_argument("--patience", type=int, default=12, help="early-stop patience")
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch, tiny subset feel — just proves the pipeline runs")
    ap.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data", "basketball"),
                    help="dataset directory containing data.yaml + train/valid/test splits")
    ap.add_argument("--name", default=None, help="run name under models/ (default: ball_rim_v1)")
    ap.add_argument("--base-weights", default="yolo11n.pt",
                    help="starting checkpoint to fine-tune from. Default is the generic "
                         "COCO-pretrained yolo11n.pt; pass a path to an existing .pt (e.g. "
                         "models/ball_detector_v2025/weights/best.pt) to fine-tune from it "
                         "instead — Ultralytics keeps the backbone and reinitializes the head "
                         "for this run's class count.")
    args = ap.parse_args()

    epochs = 1 if args.smoke else args.epochs
    name = "smoke_test" if args.smoke else (args.name or "ball_rim_v1")

    data_yaml = make_local_yaml(os.path.abspath(args.data_dir))
    print(f"Data config: {data_yaml}")
    print(f"Device: {args.device} | epochs: {epochs} | batch: {args.batch} | imgsz: {args.imgsz}")
    print(f"Base weights: {args.base_weights}")

    model = YOLO(args.base_weights)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=os.path.join(PROJECT_ROOT, "models"),
        name=name,
        patience=args.patience,
        cache=args.cache,
        amp=False,  # AMP (mixed precision) is buggy on Apple MPS — causes an
                    # intermittent bbox_iou tensor-size crash. fp32 is stable.
        exist_ok=True,
        plots=True,
    )
    weights = os.path.join(PROJECT_ROOT, "models", name, "weights", "best.pt")
    print(f"\nDone. Best weights: {weights}")


if __name__ == "__main__":
    main()
