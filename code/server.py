"""
server.py -- FastAPI backend for the Huge Highlights web UI.

Upload a video, it runs the detection pipeline in a background thread
(frame-count + ETA progress polled from the frontend), then cuts one clip
per made shot and stitches them into a single highlight reel.

Run:
    .venv/bin/uvicorn server:app --reload --port 8000
"""
import os
import sys
import time
import uuid
import shutil
import threading
import subprocess

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from simple_make import detect
from make_detector import dedup_makes
from clip_extractor import extract_all, concat_clips

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
JOBS_ROOT = os.path.join(PROJECT_ROOT, "_jobs")
os.makedirs(JOBS_ROOT, exist_ok=True)

# The 2025 tactical-analysis pipeline runs as an isolated subprocess using its
# own venv, so its pinned deps never collide with this project's.
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "analysis2025")
ANALYSIS_PY = os.path.join(ANALYSIS_DIR, ".venv", "bin", "python3")
ANALYSIS_SCRIPT = os.path.join(ANALYSIS_DIR, "run_analysis.py")

app = FastAPI()


@app.middleware("http")
async def no_cache(request: Request, call_next):
    # This is an actively-developed local tool; never let the browser cache the
    # HTML/CSS/JS or we end up debugging a stale UI (as happened during dev).
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-store"
    return resp

# job_id -> state dict, mutated by the worker thread and read by /status polls
jobs = {}


def _run_job(job_id, src_path):
    job = jobs[job_id]
    work_dir = os.path.dirname(src_path)
    frames_dir = os.path.join(work_dir, "frames")
    clips_dir = os.path.join(work_dir, "clips")
    job["start_time"] = time.time()

    def on_progress(done, total):
        job["frames_done"] = done
        job["frames_total"] = total

    try:
        job["stage"] = "detecting"
        makes, fps = detect(src_path, out_dir=frames_dir, verbose=False,
                             progress_cb=on_progress, target_fps=24)
        makes = dedup_makes(makes, min_gap_s=7.0)

        job["stage"] = "cutting"
        clip_paths = []
        reel_path = None
        side_reels = {}
        if makes:
            clip_paths = extract_all(src_path, makes, clips_dir, pre=6.0, post=2.5)
            job["stage"] = "stitching"
            if len(clip_paths) > 1:
                reel_path = concat_clips(clip_paths, os.path.join(clips_dir, "highlight_reel.mp4"))
            else:
                reel_path = clip_paths[0]

            # Per-basket reels: group makes by which hoop (left/right) they went
            # through, so each team's makes can be watched on their own. Only
            # build a side reel when that side actually has makes.
            for side in ("left", "right"):
                side_clips = [p for m, p in zip(makes, clip_paths) if m.get("side") == side]
                if not side_clips:
                    continue
                if len(side_clips) > 1:
                    out = os.path.join(clips_dir, f"reel_{side}.mp4")
                    side_reels[side] = os.path.basename(concat_clips(side_clips, out))
                else:
                    # single clip on this side: reuse the clip file itself
                    side_reels[side] = os.path.basename(side_clips[0])

        job["makes"] = [{"time": m["time"], "clip": os.path.basename(p),
                         "side": m.get("side")}
                        for m, p in zip(makes, clip_paths)]
        job["reel"] = os.path.basename(reel_path) if reel_path else None
        job["side_reels"] = side_reels
        job["stage"] = "done"
    except Exception as e:
        job["stage"] = "error"
        job["error"] = str(e)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:12]
    work_dir = os.path.join(JOBS_ROOT, job_id)
    os.makedirs(work_dir, exist_ok=True)
    src_path = os.path.join(work_dir, "source.mp4")

    with open(src_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs[job_id] = {
        "stage": "queued", "frames_done": 0, "frames_total": None,
        "makes": [], "reel": None, "side_reels": {}, "error": None,
        "start_time": None, "work_dir": work_dir,
    }
    threading.Thread(target=_run_job, args=(job_id, src_path), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    frac = (job["frames_done"] / job["frames_total"]) if job["frames_total"] else 0.0
    eta = None
    if job["start_time"] and job["frames_done"] and job["frames_total"]:
        elapsed = time.time() - job["start_time"]
        eta = (elapsed / job["frames_done"]) * (job["frames_total"] - job["frames_done"])
    return {
        "stage": job["stage"],
        "frames_done": job["frames_done"],
        "frames_total": job["frames_total"],
        "progress": frac,
        "eta_seconds": eta,
        "makes": job["makes"],
        "reel": job["reel"],
        "side_reels": job.get("side_reels", {}),
        "error": job["error"],
    }


@app.get("/api/file/{job_id}/{kind}/{filename}")
def get_file(job_id: str, kind: str, filename: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    if kind not in ("clips",):
        raise HTTPException(400, "bad kind")
    path = os.path.join(job["work_dir"], kind, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "file not found")
    return FileResponse(path, media_type="video/mp4")


# ---- 2025 tactical analysis: per-clip, on demand ----
# analyses[(job_id, clip)] -> {"stage": running|done|error, "output": name, "error": str}
analyses = {}


def _run_analysis(job_id, clip, in_path, out_path):
    key = (job_id, clip)
    try:
        proc = subprocess.run(
            [ANALYSIS_PY, ANALYSIS_SCRIPT, in_path, out_path],
            cwd=ANALYSIS_DIR, capture_output=True, text=True,
        )
        if proc.returncode == 0 and os.path.exists(out_path):
            analyses[key] = {"stage": "done", "output": os.path.basename(out_path), "error": None}
        else:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-4:]
            analyses[key] = {"stage": "error", "output": None, "error": "\n".join(tail) or "analysis failed"}
    except Exception as e:
        analyses[key] = {"stage": "error", "output": None, "error": str(e)}


@app.post("/api/analyze/{job_id}/{clip}")
def start_analysis(job_id: str, clip: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    clip = os.path.basename(clip)  # no path traversal
    in_path = os.path.join(job["work_dir"], "clips", clip)
    if not os.path.exists(in_path):
        raise HTTPException(404, "clip not found")
    if not os.path.exists(ANALYSIS_PY):
        raise HTTPException(503, "2025 analysis environment not set up")

    key = (job_id, clip)
    cur = analyses.get(key)
    if cur and cur["stage"] == "running":
        return {"stage": "running"}
    out_name = "analyzed_" + clip
    out_path = os.path.join(job["work_dir"], "clips", out_name)
    if os.path.exists(out_path):  # cached from a previous run
        analyses[key] = {"stage": "done", "output": out_name, "error": None}
        return {"stage": "done", "output": out_name}

    analyses[key] = {"stage": "running", "output": None, "error": None}
    threading.Thread(target=_run_analysis, args=(job_id, clip, in_path, out_path), daemon=True).start()
    return {"stage": "running"}


@app.get("/api/analyze_status/{job_id}/{clip}")
def analysis_status(job_id: str, clip: str):
    clip = os.path.basename(clip)
    st = analyses.get((job_id, clip))
    if st is None:
        return {"stage": "idle"}
    return st


app.mount("/", StaticFiles(directory=os.path.join(PROJECT_ROOT, "web"), html=True), name="web")
