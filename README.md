# Huge Highlights (Made Shot Highlighter)

An AI pipeline that takes raw basketball game footage and automatically detects every made shot, then cuts each one into a clip and stitches them into a single highlight reel — no manual scrubbing through hours of film required.

## The Problem

Building highlight reels or reviewing shooting performance currently means watching entire game recordings and manually marking every made shot. This is slow, doesn't scale as more games get filmed, and makes it hard for players building recruiting tapes or coaches running film sessions to quickly find what matters.

## My Solution

The system uses two independent detection models rather than one model doing everything:

1. **Ball detector** — a local YOLO model (7 classes: ball, clock, hoop, overlay, player, ref, scoreboard), run on every analyzed frame since ball tracking needs to be fast.
2. **Rim detector** — a separate RF-DETR model (90.3% mAP), run every 6 frames since rim position changes slowly, with position smoothed between runs.

Make detection uses geometry, not trajectory prediction: the detected rim box is enlarged into an L-shape — a wide zone above the rim (catching the ball's incoming arc) and a narrow zone below it (only reachable if the ball actually falls through). A shot only counts as made if the ball is detected in the above-zone for several consecutive frames, followed by a genuine detection in the below-zone. This ordering prevents false positives from balls dribbled under the hoop or bounced out to the side.

Processing runs in passes: raw detection → gap interpolation for brief ball occlusions → make-detection logic → clip extraction (ffmpeg) → reel stitching. Frame-skipping and rim-staleness checks keep processing fast enough to run locally on an M4 Pro Mac.

## Results

Validated on an 8-clip test set spanning clean broadcast footage and cluttered highlight-style montage clips:

- **7 of 8** known makes correctly identified, **zero false positives**
- Correctly rejected a bounce-out (carom) edge case
- Ball detection accuracy: **~79–87%** on clean broadcast footage, **~59%** on montage-style footage with graphics/replay cuts
- Rim detection performed reliably across the entire test set

The two missed cases both traced to ball detection failures (a phantom lock onto a stationary broadcast graphic, and a flat camera angle where the ball never rose above the rim in-frame) — confirming ball detection, not rim detection or the make-logic, is the current bottleneck.

## Repository Structure
- `/code` — Detection scripts (`simple_make.py`, `make_detector.py`), clip extraction (`clip_extractor.py`), pipeline entry point, Streamlit and FastAPI frontends
- `/docs` — Research paper draft and pitch deck
- `/data` — Dataset references
- `/results` — Validation output, thumbnails, example clips

## Dataset

- Basketball Object Detection Model (ball, person, rim) — 9,515 images — https://universe.roboflow.com/cricket-qnb5l/basketball-xil7x/dataset/1
- Basketball hoop images — 3,630 images, hoop-focused — https://universe.roboflow.com/basketball-hoop-tsdku/basketball-hoop-images/dataset/1

Model weights are excluded from this repo (100MB+, over GitHub's limit) — see `MODELS.md` for download instructions.

## Project Website

*(Add your portfolio site link once published.)*

## Contact

Hugo Nelson — *hugolinnelson@gmail.com*
