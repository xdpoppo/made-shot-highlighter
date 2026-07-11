"""
generate_rim_crops.py — augment the rim dataset with zoomed-in crops.

For every image+label pair in data/rim_only/<split>, crops a square region
~4x the rim's bbox size (centered on the rim), recomputes the YOLO label for
that crop, and saves the result to data/rim_only_crops/<split>/. Optionally
uploads each crop straight into the Roboflow project so a new dataset version
can be generated from the web UI.

The rim is often tiny in the raw broadcast frame (~2-3% of the image), so
pairing every original with a zoomed-in view teaches the model what a
close-up rim looks like, without losing the wide-shot examples.

Usage:
    python scripts/generate_rim_crops.py --limit 20               # dry run, no upload
    python scripts/generate_rim_crops.py --limit 20 --upload       # test upload
    python scripts/generate_rim_crops.py --upload                  # full run, all splits
"""
import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "rim_only")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "rim_only_crops")
SPLITS = ["train", "valid", "test"]
PAD_FACTOR = 4.0  # crop side = PAD_FACTOR * max(bbox_w, bbox_h)


def read_label(path):
    """Returns list of (cls, cx, cy, w, h) in normalized [0,1] coords."""
    boxes = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = parts
            boxes.append((int(cls), float(cx), float(cy), float(w), float(h)))
    return boxes


def crop_for_box(img_w, img_h, cx, cy, w, h, pad_factor=PAD_FACTOR):
    """Compute a square pixel crop box centered on the given normalized bbox,
    clamped inside the image. Returns (x1, y1, x2, y2) in pixels."""
    box_w_px, box_h_px = w * img_w, h * img_h
    side = max(box_w_px, box_h_px) * pad_factor
    side = min(side, img_w, img_h)  # never bigger than the image itself

    cx_px, cy_px = cx * img_w, cy * img_h
    x1 = cx_px - side / 2
    y1 = cy_px - side / 2
    x2 = x1 + side
    y2 = y1 + side

    # shift back inside bounds rather than shrinking (keeps crop square)
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > img_w:
        x1 -= (x2 - img_w)
        x2 = img_w
    if y2 > img_h:
        y1 -= (y2 - img_h)
        y2 = img_h

    return max(0, int(x1)), max(0, int(y1)), min(img_w, int(x2)), min(img_h, int(y2))


def remap_box_to_crop(cx, cy, w, h, img_w, img_h, crop_box):
    """Recompute a normalized bbox relative to the crop region."""
    x1, y1, x2, y2 = crop_box
    crop_w, crop_h = x2 - x1, y2 - y1

    cx_px, cy_px = cx * img_w, cy * img_h
    box_w_px, box_h_px = w * img_w, h * img_h

    new_cx = (cx_px - x1) / crop_w
    new_cy = (cy_px - y1) / crop_h
    new_w = box_w_px / crop_w
    new_h = box_h_px / crop_h
    return new_cx, new_cy, new_w, new_h


def process_split(split, limit=None, target_size=640):
    img_dir = os.path.join(DATASET_DIR, split, "images")
    lbl_dir = os.path.join(DATASET_DIR, split, "labels")
    out_img_dir = os.path.join(OUT_DIR, split, "images")
    out_lbl_dir = os.path.join(OUT_DIR, split, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    names = sorted(os.listdir(img_dir))
    if limit:
        names = names[:limit]

    generated = []
    for name in names:
        stem, ext = os.path.splitext(name)
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.exists(lbl_path):
            continue
        boxes = read_label(lbl_path)
        if not boxes:
            continue

        img_path = os.path.join(img_dir, name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # Use the first (usually only) rim box to center the crop.
        cls, cx, cy, w, h = boxes[0]
        crop_box = crop_for_box(img_w, img_h, cx, cy, w, h)
        x1, y1, x2, y2 = crop_box
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue  # degenerate crop, skip

        cropped = img[y1:y2, x1:x2]
        cropped = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        # Recompute labels for every box that falls (at least partly) inside the crop.
        new_lines = []
        for b_cls, b_cx, b_cy, b_w, b_h in boxes:
            b_cx_px, b_cy_px = b_cx * img_w, b_cy * img_h
            if not (x1 <= b_cx_px <= x2 and y1 <= b_cy_px <= y2):
                continue
            ncx, ncy, nw, nh = remap_box_to_crop(b_cx, b_cy, b_w, b_h, img_w, img_h, crop_box)
            nw, nh = min(nw, 1.0), min(nh, 1.0)
            new_lines.append(f"{b_cls} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")
        if not new_lines:
            continue

        out_name = f"{stem}_cropzoom{ext}"
        out_img_path = os.path.join(out_img_dir, out_name)
        out_lbl_path = os.path.join(out_lbl_dir, f"{stem}_cropzoom.txt")
        cv2.imwrite(out_img_path, cropped)
        with open(out_lbl_path, "w") as f:
            f.write("\n".join(new_lines) + "\n")

        generated.append((out_img_path, out_lbl_path))

    return generated


def upload_batch(generated, split, workspace, project_name, api_key, batch_name):
    import roboflow

    rf = roboflow.Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    labelmap = {0: "rim"}  # matches data/rim_only/data.yaml names: ['rim']

    ok, failed = 0, 0
    for img_path, lbl_path in generated:
        try:
            project.upload(
                image_path=img_path,
                annotation_path=lbl_path,
                annotation_labelmap=labelmap,
                split=split,
                batch_name=batch_name,
            )
            ok += 1
        except Exception as e:
            failed += 1
            print(f"  FAILED: {os.path.basename(img_path)} -> {e}")
    print(f"  [{split}] uploaded {ok} ok, {failed} failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                     help="cap number of source images per split (for a quick test run)")
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--upload", action="store_true", help="upload generated crops to Roboflow")
    ap.add_argument("--batch-name", default="cropzoom-augmentation", dest="batch_name")
    args = ap.parse_args()

    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    workspace = os.environ.get("ROBOFLOW_WORKSPACE")
    project_name = os.environ.get("ROBOFLOW_PROJECT")

    if args.upload and not (api_key and workspace and project_name):
        raise SystemExit("Missing ROBOFLOW_API_KEY / ROBOFLOW_WORKSPACE / ROBOFLOW_PROJECT in .env")

    total = 0
    for split in args.splits:
        print(f"Generating crops for split '{split}' ...")
        generated = process_split(split, limit=args.limit)
        print(f"  generated {len(generated)} crop(s) -> {os.path.join(OUT_DIR, split)}")
        total += len(generated)

        if args.upload and generated:
            print(f"  uploading {len(generated)} crop(s) to Roboflow project '{project_name}' ...")
            upload_batch(generated, split, workspace, project_name, api_key, args.batch_name)

    print(f"\nDone. {total} crop(s) generated across splits: {args.splits}")


if __name__ == "__main__":
    main()
