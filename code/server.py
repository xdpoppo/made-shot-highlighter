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

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from simple_make import detect
from make_detector import dedup_makes
from clip_extractor import extract_all, concat_clips

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
JOBS_ROOT = os.path.join(PROJECT_ROOT, "_jobs")
os.makedirs(JOBS_ROOT, exist_ok=True)

app = FastAPI()

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
        if makes:
            clip_paths = extract_all(src_path, makes, clips_dir, pre=6.0, post=2.5)
            job["stage"] = "stitching"
            if len(clip_paths) > 1:
                reel_path = concat_clips(clip_paths, os.path.join(clips_dir, "highlight_reel.mp4"))
            else:
                reel_path = clip_paths[0]

        job["makes"] = [{"time": m["time"], "clip": os.path.basename(p)}
                         for m, p in zip(makes, clip_paths)]
        job["reel"] = os.path.basename(reel_path) if reel_path else None
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
        "makes": [], "reel": None, "error": None, "start_time": None,
        "work_dir": work_dir,
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


app.mount("/", StaticFiles(directory=os.path.join(PROJECT_ROOT, "web"), html=True), name="web")
