"""
verify_dataset.py — Phase 2 gate for the Made Shot Highlighter.

Inspects a YOLO-format detection dataset BEFORE we train on it, to catch the
failure mode we hit once already: a "classification" dataset where every box
covers the whole frame (useless for localization/tracking).

Usage:
    python scripts/verify_dataset.py --zip ~/Desktop/Basketball.v1i.yolov11.zip --name basketball
    python scripts/verify_dataset.py --dir data/basketball          # already-extracted

What it does:
    1. Extracts the zip into data/<name>/ (if --zip given).
    2. Prints classes from data.yaml.
    3. Counts images/labels per split and the class-instance distribution.
    4. Reports box-size stats (flags full-frame / >90% boxes).
    5. Renders a few annotated sample images to output/dataset_check/.
"""
import argparse
import glob
import os
import zipfile
from collections import Counter

import cv2
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def extract(zip_path, name):
    dest = os.path.join(PROJECT_ROOT, "data", name)
    os.makedirs(dest, exist_ok=True)
    print(f"Extracting {zip_path}\n       -> {dest} ...")
    with zipfile.ZipFile(os.path.expanduser(zip_path)) as z:
        z.extractall(dest)
    return dest


def load_yaml(dataset_dir):
    yml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(yml_path):
        raise FileNotFoundError(f"No data.yaml in {dataset_dir}")
    with open(yml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    # names may be a list or a dict {0: 'ball', ...}
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return data, names


def find_split_dir(dataset_dir, split):
    """Roboflow layout is usually dataset_dir/<split>/images + /labels."""
    cand = os.path.join(dataset_dir, split, "labels")
    if os.path.isdir(cand):
        return os.path.join(dataset_dir, split, "images"), cand
    return None, None


def image_for_label(images_dir, stem):
    for ext in IMG_EXTS:
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def analyze_split(images_dir, labels_dir, names):
    cls = Counter()
    boxes_per_img = Counter()
    total_boxes = fullframe = big = 0
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    for f in label_files:
        with open(f) as fh:
            rows = [ln.split() for ln in fh.read().split("\n") if ln.strip()]
        boxes_per_img[len(rows)] += 1
        for p in rows:
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            w, h = float(p[3]), float(p[4])
            cls[c] += 1
            total_boxes += 1
            if w >= 0.98 and h >= 0.98:
                fullframe += 1
            if w * h >= 0.9:
                big += 1
    return {
        "n_labels": len(label_files),
        "n_images": len([p for p in glob.glob(os.path.join(images_dir, "*")) if p.lower().endswith(IMG_EXTS)]) if images_dir else 0,
        "cls": cls,
        "boxes_per_img": boxes_per_img,
        "total_boxes": total_boxes,
        "fullframe": fullframe,
        "big": big,
    }


def render_samples(images_dir, labels_dir, names, out_dir, per_class=1):
    os.makedirs(out_dir, exist_ok=True)
    want = {}  # class_id -> needed count
    seen = Counter()
    saved = []
    for f in glob.glob(os.path.join(labels_dir, "*.txt")):
        with open(f) as fh:
            rows = [ln.split() for ln in fh.read().split("\n") if ln.strip()]
        if not rows:
            continue
        classes_here = {int(float(r[0])) for r in rows if len(r) >= 5}
        target = next((c for c in classes_here if seen[c] < per_class), None)
        if target is None:
            continue
        stem = os.path.splitext(os.path.basename(f))[0]
        img_path = image_for_label(images_dir, stem)
        if not img_path:
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        for r in rows:
            if len(r) < 5:
                continue
            c = int(float(r[0]))
            xc, yc, w, h = map(float, r[1:5])
            x1, y1 = int((xc - w / 2) * W), int((yc - h / 2) * H)
            x2, y2 = int((xc + w / 2) * W), int((yc + h / 2) * H)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = names[c] if c < len(names) else str(c)
            cv2.putText(img, label, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        out = os.path.join(out_dir, f"{names[target] if target < len(names) else target}_{seen[target]}.jpg")
        cv2.imwrite(out, img)
        saved.append(out)
        seen[target] += 1
        if all(seen[c] >= per_class for c in range(len(names))):
            break
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", help="path to dataset zip")
    ap.add_argument("--dir", help="path to already-extracted dataset dir")
    ap.add_argument("--name", default="dataset", help="folder name under data/ when extracting")
    args = ap.parse_args()

    if args.zip:
        dataset_dir = extract(args.zip, args.name)
    elif args.dir:
        dataset_dir = args.dir if os.path.isabs(args.dir) else os.path.join(PROJECT_ROOT, args.dir)
    else:
        ap.error("provide --zip or --dir")

    # Roboflow zips sometimes nest one level; find the dir containing data.yaml
    if not os.path.exists(os.path.join(dataset_dir, "data.yaml")):
        hits = glob.glob(os.path.join(dataset_dir, "**", "data.yaml"), recursive=True)
        if hits:
            dataset_dir = os.path.dirname(hits[0])

    data, names = load_yaml(dataset_dir)
    print("\n" + "=" * 60)
    print(f"Dataset: {dataset_dir}")
    print(f"Classes (nc={data.get('nc')}): {names}")
    print("=" * 60)

    grand = Counter()
    grand_total = grand_full = grand_big = 0
    for split in ("train", "valid", "val", "test"):
        images_dir, labels_dir = find_split_dir(dataset_dir, split)
        if not labels_dir:
            continue
        s = analyze_split(images_dir, labels_dir, names)
        if s["total_boxes"] == 0:
            continue
        print(f"\n--- {split} ---")
        print(f"  images: {s['n_images']}   labels: {s['n_labels']}   boxes: {s['total_boxes']}")
        print(f"  boxes/image: " + ", ".join(f"{k}:{v}" for k, v in sorted(s['boxes_per_img'].items())))
        print("  class distribution:")
        for c in sorted(s["cls"]):
            nm = names[c] if c < len(names) else str(c)
            print(f"      {c} {nm:12s}: {s['cls'][c]}")
        ff = 100 * s["fullframe"] / s["total_boxes"]
        bg = 100 * s["big"] / s["total_boxes"]
        print(f"  full-frame boxes (w&h>=0.98): {s['fullframe']} ({ff:.1f}%)")
        print(f"  boxes >=90% of frame:         {s['big']} ({bg:.1f}%)")
        grand += s["cls"]
        grand_total += s["total_boxes"]
        grand_full += s["fullframe"]
        grand_big += s["big"]

    # Render samples from the train split
    tr_images, tr_labels = find_split_dir(dataset_dir, "train")
    out_dir = os.path.join(PROJECT_ROOT, "output", "dataset_check")
    saved = render_samples(tr_images, tr_labels, names, out_dir, per_class=1) if tr_labels else []

    # ---- verdict ----
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    ff_pct = 100 * grand_full / grand_total if grand_total else 0
    big_pct = 100 * grand_big / grand_total if grand_total else 0
    lname = [n.lower() for n in names]
    has_ball = any("ball" in n for n in lname)
    has_rim = any(("rim" in n) or ("hoop" in n) or ("basket" in n and "ball" not in n) for n in lname)
    print(f"  total boxes: {grand_total}")
    print(f"  full-frame boxes overall: {ff_pct:.1f}%   (>=90%: {big_pct:.1f}%)")
    print(f"  has a ball class: {has_ball}")
    print(f"  has a rim/hoop class: {has_rim}")
    localized_ok = big_pct < 50  # real detection data: most boxes are small
    print(f"  localized-boxes check (<50% big): {'PASS' if localized_ok else 'FAIL'}")
    if saved:
        print(f"  sample renders ({len(saved)}): {out_dir}")
    ok = localized_ok and has_ball and has_rim
    print("\n  >>> " + ("LOOKS GOOD — real detection data with ball + rim. Proceed to training."
                        if ok else
                        "PROBLEM — see flags above. Do NOT train until resolved."))
    print("=" * 60)


if __name__ == "__main__":
    main()
