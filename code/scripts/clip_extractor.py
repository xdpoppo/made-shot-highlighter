"""
clip_extractor.py — Phase 6: cut a short clip around each made shot.

Prefers ffmpeg (fast, keeps audio); falls back to OpenCV frame copying (no audio)
if ffmpeg isn't installed, so the tool works with zero extra system deps.

Usage (standalone):
    python scripts/clip_extractor.py --source game.mp4 --time 42.5 --out output/clips/make_001.mp4
"""
import os
import shutil
import subprocess

_FFMPEG = shutil.which("ffmpeg")
if not _FFMPEG:
    try:
        import imageio_ffmpeg
        _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass


def extract_clip(source, t_center, out_path, pre=3.0, post=3.0):
    """Cut [t_center - pre, t_center + post] seconds from source into out_path."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    start = max(0.0, t_center - pre)
    dur = pre + post

    if _FFMPEG:
        cmd = [_FFMPEG, "-y", "-i", source, "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
               "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac",
               "-loglevel", "error", out_path]
        try:
            subprocess.run(cmd, check=True)
            return out_path
        except subprocess.CalledProcessError:
            pass  # fall through to OpenCV

    _extract_opencv(source, start, dur, out_path)
    return out_path


def _extract_opencv(source, start, dur, out_path):
    import cv2
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_f = int(start * fps)
    end_f = int((start + dur) * fps)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    f = start_f
    while f < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        f += 1
    cap.release()
    writer.release()


def extract_all(source, makes, out_dir, pre=3.0, post=3.0):
    """Extract one clip per make event. `makes` is a list of {'time': seconds}."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, m in enumerate(makes, 1):
        out = os.path.join(out_dir, f"make_{i:03d}.mp4")
        extract_clip(source, m["time"], out, pre=pre, post=post)
        paths.append(out)
    return paths


def _has_audio(path):
    if not _FFMPEG:
        return False
    probe = subprocess.run(
        [_FFMPEG, "-i", path], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    return "Audio:" in probe.stderr


def concat_clips(clip_paths, out_path):
    """Stitch already-cut clips into one highlight reel, in order."""
    if not clip_paths:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    if _FFMPEG:
        # Clips cut from a source with no audio track are video-only; mapping a
        # nonexistent [i:a:0] stream makes ffmpeg exit nonzero, so check first.
        has_audio = all(_has_audio(p) for p in clip_paths)
        n = len(clip_paths)
        inputs = []
        for p in clip_paths:
            inputs += ["-i", p]
        if has_audio:
            filter_complex = "".join(f"[{i}:v:0][{i}:a:0]" for i in range(n)) + f"concat=n={n}:v=1:a=1[outv][outa]"
            cmd = [_FFMPEG, "-y", *inputs, "-filter_complex", filter_complex,
                   "-map", "[outv]", "-map", "[outa]",
                   "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac",
                   "-loglevel", "error", out_path]
        else:
            filter_complex = "".join(f"[{i}:v:0]" for i in range(n)) + f"concat=n={n}:v=1:a=0[outv]"
            cmd = [_FFMPEG, "-y", *inputs, "-filter_complex", filter_complex,
                   "-map", "[outv]",
                   "-c:v", "libx264", "-preset", "veryfast",
                   "-loglevel", "error", out_path]
        subprocess.run(cmd, check=True)
        return out_path

    _concat_opencv(clip_paths, out_path)
    return out_path


def _concat_opencv(clip_paths, out_path):
    import cv2
    cap0 = cv2.VideoCapture(clip_paths[0])
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for p in clip_paths:
        cap = cv2.VideoCapture(p)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        cap.release()
    writer.release()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--time", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pre", type=float, default=3.0)
    ap.add_argument("--post", type=float, default=3.0)
    args = ap.parse_args()
    p = extract_clip(args.source, args.time, args.out, pre=args.pre, post=args.post)
    print(f"Wrote {p}  (ffmpeg={'yes' if _FFMPEG else 'no, used OpenCV'})")
