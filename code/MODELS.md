## Model weights

Weight files are not committed to this repo (100MB+ each, over GitHub's limit).

- **Ball detector** — YOLOv5l6u, `models/ball_detector_v2025/weights/best.pt`. Trained on a labeled basketball dataset (ball/clock/hoop/overlay/player/ref/scoreboard classes).
- **Rim detector** — RF-DETR Small, `models/rf_detr_rim_v1/weights.pt`. Fine-tuned on a Roboflow rim dataset (90.3% mAP).

Both are loaded locally by `scripts/simple_make.py` at the paths above. To reproduce, retrain with `scripts/train.py` (ball) and the RF-DETR fine-tuning flow referenced in `scripts/merge_rim_dataset.py` / `scripts/generate_rim_crops.py`, or contact the author for the trained checkpoints.
