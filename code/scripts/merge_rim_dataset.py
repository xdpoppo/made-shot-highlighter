"""
merge_rim_dataset.py — combine the extra hoop dataset into data/basketball.

Adds more `ball` and `rim` examples to strengthen detection. Source classes are
remapped to the destination's class ids and files are copied in with a prefix so
nothing collides and the merge is identifiable/reversible.

    data/hoop_extra classes:   0=basketball, 1=rim
    data/basketball classes:   0=ball, 1=human, 2=rim
    class map:                 0 -> 0 (ball),  1 -> 2 (rim)

Usage:
    python scripts/merge_rim_dataset.py
    python scripts/merge_rim_dataset.py --undo     # remove previously merged files
"""
import argparse
import glob
import os
import re
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(PROJECT_ROOT, "data", "hoop_extra")
DST = os.path.join(PROJECT_ROOT, "data", "basketball")
PREFIX = "hoopx_"
CLASS_MAP = {0: 0, 1: 2}  # source id -> dest id
SPLIT_MAP = {"train": "train", "valid": "valid", "test": "test"}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def undo():
    removed = 0
    for split in SPLIT_MAP.values():
        for sub in ("images", "labels"):
            for p in glob.glob(os.path.join(DST, split, sub, PREFIX + "*")):
                os.remove(p)
                removed += 1
    print(f"Removed {removed} previously merged files.")


def remap_label(src_path, dst_path):
    """Copy a label file, translating class ids; drop classes not in CLASS_MAP."""
    out_lines = []
    with open(src_path) as f:
        for ln in f.read().split("\n"):
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            cid = int(float(parts[0]))
            if cid not in CLASS_MAP:
                continue
            parts[0] = str(CLASS_MAP[cid])
            out_lines.append(" ".join(parts))
    with open(dst_path, "w") as f:
        f.write("\n".join(out_lines) + ("\n" if out_lines else ""))


def _source_prefix(stem):
    """Strip Roboflow's `.rf.<hash>` so augmented copies of one source group together."""
    return re.sub(r"\.rf\.[a-f0-9]+$", "", stem)


def _aug_score(img_path):
    """Higher = more augmented. Counts near-black corner pixels (rotation/shear fill)."""
    import cv2
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 1e9
    h, w = img.shape[:2]
    c = max(8, min(h, w) // 20)
    corners = [img[:c, :c], img[:c, -c:], img[-c:, :c], img[-c:, -c:]]
    return int(sum((corner < 10).sum() for corner in corners))


def _select(src_labels, src_images, dedup):
    """Return list of (label_path, image_path) to merge; if dedup, 1 per source image."""
    pairs = []
    for lbl in glob.glob(os.path.join(src_labels, "*.txt")):
        stem = os.path.splitext(os.path.basename(lbl))[0]
        img = next((os.path.join(src_images, stem + e)
                    for e in IMG_EXTS if os.path.exists(os.path.join(src_images, stem + e))), None)
        if img is not None:
            pairs.append((lbl, img))
    if not dedup:
        return pairs
    # keep the least-augmented copy per source prefix
    groups = {}
    for lbl, img in pairs:
        key = _source_prefix(os.path.splitext(os.path.basename(lbl))[0])
        groups.setdefault(key, []).append((lbl, img))
    kept = []
    for key, members in groups.items():
        best = min(members, key=lambda p: _aug_score(p[1])) if len(members) > 1 else members[0]
        kept.append(best)
    return kept


def merge(dedup=True):
    copied = 0
    per_split = {}
    for src_split, dst_split in SPLIT_MAP.items():
        src_labels = os.path.join(SRC, src_split, "labels")
        src_images = os.path.join(SRC, src_split, "images")
        if not os.path.isdir(src_labels):
            continue
        dst_labels = os.path.join(DST, dst_split, "labels")
        dst_images = os.path.join(DST, dst_split, "images")
        os.makedirs(dst_labels, exist_ok=True)
        os.makedirs(dst_images, exist_ok=True)

        n = 0
        for lbl, img in _select(src_labels, src_images, dedup):
            stem = os.path.splitext(os.path.basename(lbl))[0]
            remap_label(lbl, os.path.join(dst_labels, PREFIX + stem + ".txt"))
            shutil.copy2(img, os.path.join(dst_images, PREFIX + os.path.basename(img)))
            n += 1
            copied += 1
        per_split[dst_split] = n
    print(f"Merged files per split ({'de-augmented' if dedup else 'all'}):", per_split)
    print(f"Total added: {copied}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--undo", action="store_true")
    ap.add_argument("--keep-augmented", action="store_true",
                    help="merge ALL images incl. augmented copies (default: de-augment, 1 per source)")
    args = ap.parse_args()
    if args.undo:
        undo()
    else:
        undo()   # clear any prior merge first (idempotent)
        merge(dedup=not args.keep_augmented)


if __name__ == "__main__":
    main()
