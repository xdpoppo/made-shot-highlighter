# 2025 Tactical Analysis (integrated)

Last year's Basketball Video Analysis pipeline, vendored so the current app can
run a **tactical breakdown** on any detected made-shot clip: player + ball
tracking, team assignment (CLIP on jerseys), ball possession, passes /
interceptions, a top-down tactical court map, and per-player speed & distance.

It runs as an **isolated subprocess** (`run_analysis.py`) using its own venv
(`analysis2025/.venv`) so its pinned dependencies never collide with the current
project's. The Huge Highlights server calls it per clip via `POST /api/analyze`.

## Setup (not committed — regenerate locally)

**1. Model weights** → `analysis2025/models/` (each 100MB+, gitignored):
- `player_detector.pt` — YOLO player detector
- `ball_detector_model.pt` — YOLO ball detector (identical to the current
  project's `models/ball_detector_v2025/weights/best.pt`)
- `court_keypoint_detector.pt` — YOLO-pose court keypoint model

**2. Isolated venv:**
```bash
cd analysis2025
python3 -m venv .venv
.venv/bin/pip install numpy<2 opencv-python pandas roboflow \
    supervision==0.25.1 torch transformers==4.46.3 ultralytics==8.3.67 imageio-ffmpeg
```

On first run the team-assigner downloads `patrickjohncyh/fashion-clip` from
Hugging Face (~600MB, one time).

## Run standalone
```bash
.venv/bin/python run_analysis.py <input_clip.mp4> <output.mp4>
```
Detection runs on Apple MPS (GPU) where available; unsupported ops fall back to
CPU. Output is browser-playable H.264.
