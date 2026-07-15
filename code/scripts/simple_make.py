"""
simple_make.py — made-shot detection via an enlarged rim zone + above->below.

Idea (kept deliberately simple):
  * The trained detector's rim box is small and usually only catches the BACK of
    the ring, so we ENLARGE it: horizontally toward court-center (the front of
    the rim, which the box misses), and vertically both up and down (~double
    height). The up part is the "ABOVE the rim" zone, the down part is the
    "BELOW the rim" zone; the detected ring sits between them.
  * A make requires the ball to be seen ABOVE the rim and then, within a short
    window, BELOW it — proving it actually dropped THROUGH the hoop.
  * Because we require ABOVE first, extending the box downward is safe: a ball
    merely dribbled under the basket never had an "above" frame, so it can't
    trigger a make.

Which way to extend horizontally depends on which basket: a hoop on the right
half of the frame has its front (open) side to the LEFT (toward center), and
vice-versa. We pick the direction from the rim's x-position in the frame.

Usage:
    python scripts/simple_make.py --source ~/Desktop/TestClips/broadcast_lal_vs_gs_2makes.mp4
    python scripts/simple_make.py --source clip.mp4 --out output/simple_makes
"""
import argparse
import os
from collections import deque

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from make_detector import _pick_centers

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "models", "ball_detector_v2025", "weights", "best.pt")


RFDETR_RIM_CLASS_ID = 2   # trained checkpoint's class_id for "rim" (1="ball")
RFDETR_WEIGHTS = os.path.join(PROJECT_ROOT, "models", "rf_detr_rim_v1", "weights.pt")


def _load_rfdetr_rim_model(weights=RFDETR_WEIGHTS):
    """Loads the locally-downloaded RF-DETR checkpoint (trained on the
    cropzoom-augmented rim+ball dataset) for local inference — no network
    calls. Used only for the rim class here; local ball detection stays on
    the separate, faster local YOLO model."""
    from rfdetr import RFDETRSmall
    return RFDETRSmall(pretrain_weights=weights, num_classes=2)


def _rfdetr_rim_box(rf_model, frame, conf=0.4):
    """One local-inference call -> best rim box [x1,y1,x2,y2], or None."""
    import cv2
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = rf_model.predict(img_rgb, threshold=conf)
    rim_mask = dets.class_id == RFDETR_RIM_CLASS_ID
    if not rim_mask.any():
        return None
    idx = dets.confidence[rim_mask].argmax()
    x1, y1, x2, y2 = dets.xyxy[rim_mask][idx]
    return [float(x1), float(y1), float(x2), float(y2)]


def enlarge_rim(rim, frame_w, below_factor=0.4, up_factor=0.7, down_factor=2.0,
                margin=0.3, above_factor=1.8):
    """Grow a detected rim dict into an L-shaped full-hoop zone.

    rim: dict(left, top, right, bottom). Returns a dict:
        {"above": (ax1, ay1, ax2, ay2), "below": (bx1, by1, bx2, by2)}

    The zone is L-shaped: a WIDE top bar (the "above" lane) and a NARROW stem
    under the ring (the "below" lane).

    * ABOVE is stretched far toward the side the ring opens to (frame-center),
      by `above_factor`, because a layup arrives nearly horizontally and is
      still well off to the open side while it climbs over the front of the
      rim — a tight span would miss that approach sample.
    * BELOW is kept TIGHT: only `below_factor` of a rim-width toward the open
      side (enough to cover the true ring opening the detector under-draws,
      since the box tends to catch only the back of the ring) and a small
      `margin` on the backboard side. The ball has to fall through the actual
      ring opening to count; a wide "below" lets a shot that bounces out to the
      side reappear inside the zone and register as a fake fall-through — which
      is exactly the false positive this tight stem prevents.
    Vertical growth is up by `up_factor` (above) and down by `down_factor`
    (below), both as multiples of the ring height.
    """
    rx1, ry1, rx2, ry2 = rim["left"], rim["top"], rim["right"], rim["bottom"]
    w = rx2 - rx1
    h = ry2 - ry1
    cx = (rx1 + rx2) / 2
    ay1 = ry1 - up_factor * h        # top of the ABOVE bar
    by2 = ry2 + down_factor * h      # bottom of the BELOW stem
    if cx > frame_w / 2:      # right-side hoop -> open side is to the LEFT
        bx1, bx2 = rx1 - below_factor * w, rx2 + margin * w
        ax1, ax2 = rx1 - above_factor * w, rx2 + margin * w
    else:                     # left-side hoop  -> open side is to the RIGHT
        bx1, bx2 = rx1 - margin * w, rx2 + below_factor * w
        ax1, ax2 = rx1 - margin * w, rx2 + above_factor * w
    return {
        "above": (ax1, ay1, ax2, ry1),   # wide bar from ay1 down to the ring top
        "below": (bx1, ry2, bx2, by2),   # narrow stem from ring bottom down to by2
    }


def annotate(frame, ball_box, rim, zone, t, frame_w):
    import cv2
    f = frame.copy()
    rx1, ry1, rx2, ry2 = map(int, (rim["left"], rim["top"], rim["right"], rim["bottom"]))
    ax1, ay1, ax2, ay2 = map(int, zone["above"])
    bx1, by1, bx2, by2 = map(int, zone["below"])
    # L-shaped zone: wide ABOVE bar + narrow BELOW stem, both green.
    cv2.rectangle(f, (ax1, ay1), (ax2, ay2), (0, 255, 0), 3)      # ABOVE bar
    cv2.rectangle(f, (bx1, by1), (bx2, by2), (0, 255, 0), 3)      # BELOW stem
    cv2.rectangle(f, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)      # detected ring: red
    cv2.putText(f, "ABOVE", (ax1 + 4, ay1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(f, "BELOW", (bx1 + 4, by2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if ball_box is not None:
        bx1, by1, bx2, by2 = map(int, ball_box)
        cv2.rectangle(f, (bx1, by1), (bx2, by2), (0, 220, 255), 3)  # ball: yellow
        cv2.putText(f, "BALL", (bx1, max(20, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    cv2.putText(f, f"MAKE  t={t:.2f}s", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 4)
    return f


def _interpolate_boxes(boxes, max_gap):
    """Linearly fill gaps of <= `max_gap` missing frames between two known ball
    boxes, so a ball occluded through the net/players keeps a continuous arc.
    Longer gaps (camera panned to the far end) are left empty on purpose — we
    never want to invent a through-the-rim trajectory across a real absence.
    Leading gaps (before the first detection) are not extrapolated.
    """
    out = list(boxes)
    known = [i for i, b in enumerate(out) if b is not None]
    for a, c in zip(known, known[1:]):
        gap = c - a
        if gap <= 1 or (gap - 1) > max_gap:
            continue
        ba, bc = out[a], out[c]
        for k in range(1, gap):
            t = k / gap
            out[a + k] = [ba[j] + (bc[j] - ba[j]) * t for j in range(4)]
    return out


def _smooth_rim(boxes, alpha, max_hold):
    """Per-frame rim dict from raw rim boxes: EMA-smooth on each fresh sighting,
    hold the last position through short gaps (up to `max_hold` frames), then go
    None once the rim has been unseen too long (camera moved on). Mirrors the
    old inline EMA + rim_fresh staleness check, precomputed for the whole clip.
    """
    out = [None] * len(boxes)
    rim = None
    last_seen = -10 ** 9
    for i, b in enumerate(boxes):
        if b is not None:
            if rim is None:
                rim = dict(left=b[0], top=b[1], right=b[2], bottom=b[3])
            else:
                rim = dict(
                    left=(1 - alpha) * rim["left"] + alpha * b[0],
                    top=(1 - alpha) * rim["top"] + alpha * b[1],
                    right=(1 - alpha) * rim["right"] + alpha * b[2],
                    bottom=(1 - alpha) * rim["bottom"] + alpha * b[3],
                )
            last_seen = i
        if rim is not None and (i - last_seen) <= max_hold:
            out[i] = dict(rim)
    return out


def detect(source, weights=DEFAULT_WEIGHTS, conf=0.12, device="mps", out_dir=None,
           cooldown_s=1.5, gap_s=0.8, min_above=3, rim_alpha=0.15, rim_max_stale_s=2.0,
           below_factor=0.4, up_factor=0.7, down_factor=2.0, above_factor=1.8,
           interp_gap_s=0.5, imgsz=None, verbose=True,
           rim_source="rfdetr", rim_weights=RFDETR_WEIGHTS, rim_stride=6, rim_conf=0.4,
           progress_cb=None, target_fps=None):
    """progress_cb, if given, is called periodically as progress_cb(frames_done, total_frames)
    during the (slow) per-frame detection pass, so a caller can show a progress bar/ETA.

    target_fps, if given, decimates the video to roughly that analysis rate (e.g. 24) by
    skipping frames with a cheap cap.grab() (no decode/inference) instead of running the
    detectors on every native frame — the main lever for wall-clock speed on high-fps
    (60fps) source video. None (default) analyzes every frame, unchanged from before."""
    import cv2
    from ultralytics import YOLO

    model = YOLO(weights)
    names = model.names
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    stride = max(1, round(fps / target_fps)) if target_fps else 1
    effective_fps = fps / stride
    total_frames = (native_total // stride) if native_total else None
    progress_every = max(1, (total_frames or 1000) // 200)  # ~200 callbacks over the whole video
    cooldown_frames = cooldown_s * effective_fps
    gap_frames = gap_s * effective_fps
    rim_max_stale = rim_max_stale_s * effective_fps

    rf_model = None
    if rim_source == "rfdetr":
        rf_model = _load_rfdetr_rim_model(rim_weights)
        if verbose:
            print(f"  (rim source: local RF-DETR, every {rim_stride} frames)")

    # Inference resolution. The default 640 shrinks a high-res frame so much that
    # the ball becomes a few ambiguous pixels and the model fires phantom "ball"
    # boxes on heads/shoes. Scale imgsz up with the source so the ball stays
    # legible (e.g. a 3024px clip needs ~1280 to detect cleanly).
    if imgsz is None:
        imgsz = 1280 if W >= 2000 else 640
    if verbose:
        print(f"  (source width {W}px -> inference imgsz={imgsz})")

    # ---- Pass 1: raw per-frame ball + rim detection (no make logic yet) ----
    # We collect the whole clip's detections first so we can clean and
    # gap-fill the ball trajectory before judging makes — the tracking recipe
    # ported from the 2025-Hugo-Basketball-Analysis BallTracker.
    ball_hist = deque(maxlen=90)
    ball_raw = []     # per sample: [x1,y1,x2,y2] or None
    rim_raw = []      # per sample: [x1,y1,x2,y2] or None
    frame_no_seq = []  # native frame number each sample corresponds to (for accurate timing)

    frame_period = 1.0 / effective_fps  # seconds per analyzed frame
    last_rim_frame = -1e9  # frame index where rim was last detected
    rim_skip_threshold_s = 3.0  # skip ball inference if rim absent longer than this

    idx = 0
    frame_no = 0
    skip = stride - 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Cheap rim-only check every rim_stride frames (rfdetr mode). This also
        # tells us whether the ball model is even worth running: with no rim in
        # view a make is impossible anyway (pass 2 requires rim is not None),
        # so once the rim's been gone a while (dead time: timeouts, replays,
        # crowd shots) skip the expensive ball YOLO until it's back.
        skip_ball = False
        if rf_model is not None:
            if idx % rim_stride == 0:
                rim_box = _rfdetr_rim_box(rf_model, frame, rim_conf)
                if rim_box is not None:
                    last_rim_frame = idx
            else:
                rim_box = None
            rim_absent_s = (idx - last_rim_frame) * frame_period
            skip_ball = rim_absent_s > rim_skip_threshold_s
        else:
            rim_box = None  # filled in from the ball model's own rim class below

        if skip_ball:
            ball_box = None
        else:
            r = model.predict(source=frame, conf=conf, device=device, verbose=False, imgsz=imgsz)[0]
            if len(ball_hist) >= 2:
                (lf, x1, y1), (_, x0, y0) = ball_hist[-1], ball_hist[-2]
                pred = (x1 + (x1 - x0), y1 + (y1 - y0))
            elif ball_hist:
                lf, x1, y1 = ball_hist[-1]
                pred = (x1, y1)
            else:
                lf, pred = idx, None
            gap = max(1, idx - lf)
            dyn = min(600, 100 + 60 * gap)
            ball_box, local_rim_box = _pick_centers(r.boxes, names, prev_ball=pred,
                                                    max_jump=dyn, ball_hist=ball_hist)
            if rf_model is None:
                rim_box = local_rim_box

        if ball_box is not None:
            cx = (ball_box[0] + ball_box[2]) / 2
            cy = (ball_box[1] + ball_box[3]) / 2
            ball_hist.append((idx, cx, cy))
        ball_raw.append(list(ball_box) if ball_box is not None else None)
        rim_raw.append(list(rim_box) if rim_box is not None else None)
        frame_no_seq.append(frame_no)
        idx += 1
        if progress_cb is not None and (idx % progress_every == 0 or idx == total_frames):
            progress_cb(idx, total_frames)

        frame_no += 1
        for _ in range(skip):
            if not cap.grab():
                break
            frame_no += 1

    cap.release()
    n = idx

    # ---- Interpolate the ball trajectory over short occlusion gaps ----
    # NB: we do NOT run a separate remove-wrong-detections pass here. _pick_centers
    # already gates on continuity/velocity per frame; layering the reference's
    # simpler jump filter on top backfires — after an occlusion it can anchor to
    # a phantom (a ref, a shoe) and then delete the *real* ball for being "far"
    # from that phantom, destroying the very below-rim sample a make needs.
    max_gap = int(round(interp_gap_s * effective_fps))  # only bridge short occlusions
    ball_seq = _interpolate_boxes(ball_raw, max_gap)
    rim_seq = _smooth_rim(rim_raw, rim_alpha, int(round(rim_max_stale_s * effective_fps)))

    # ---- Pass 2: make logic over the smoothed trajectory ----
    above_frames = deque()   # recent frame indices where the ball was ABOVE the rim
    last_make_frame = -1e9
    makes = []
    for i in range(n):
        while above_frames and (i - above_frames[0]) > gap_frames:
            above_frames.popleft()
        rim = rim_seq[i]
        bb = ball_seq[i]
        if rim is None or bb is None:
            continue
        cx = (bb[0] + bb[2]) / 2
        cy = (bb[1] + bb[3]) / 2
        zone = enlarge_rim(rim, W, below_factor, up_factor, down_factor,
                           above_factor=above_factor)
        ax1, ay1, ax2, ay2 = zone["above"]
        bx1, by1, bx2, by2 = zone["below"]
        # L-shape: the ABOVE lane is wide (catches horizontal layups), the
        # BELOW stem is narrow (ball must fall through the ring opening).
        is_above = (ax1 <= cx <= ax2) and cy < rim["top"]
        # A linearly-interpolated point is a straight-line guess across an
        # occlusion — fine for counting sustained ABOVE presence, but the
        # final contact-confirming sample must be something the model actually
        # saw. Otherwise interpolation can draw a fabricated path back through
        # the BELOW stem for a ball that really caromed out to the side.
        is_below = (ball_raw[i] is not None) \
            and (bx1 <= cx <= bx2) and rim["bottom"] < cy <= by2

        if is_above:
            above_frames.append(i)
        elif is_below and len(above_frames) >= min_above \
                and (i - last_make_frame) >= cooldown_frames:
            # Sustained ABOVE presence (a real approach) then a drop to BELOW
            # = the ball passed through. A 1-2 frame "above" blip (a phantom
            # detection on the backboard/sign) can't clear min_above, so it
            # won't pair with an unrelated "below" detection into a fake make.
            last_make_frame = i
            above_frames.clear()
            native_frame = frame_no_seq[i]
            t = native_frame / fps
            # Which basket did this go through? The hoop being attacked sits on
            # one side of the frame; classify by the rim center's x vs the frame
            # midline. Within a period each team attacks a fixed basket, so
            # "left" vs "right" separates the two teams' makes. (Teams switch
            # ends at halftime, so a full game needs a per-half side->team map.)
            rim_cx = (rim["left"] + rim["right"]) / 2
            side = "left" if rim_cx < W / 2 else "right"
            makes.append({"frame": native_frame, "time": t, "zone": zone,
                          "ball": bb, "rim": rim,
                          "side": side, "rim_cx": rim_cx, "frame_w": W})
            if verbose:
                print(f"  MAKE @ frame {native_frame} (t={t:.2f}s) [{side} basket]")

    # ---- Pass 3: re-read only the make frames to save annotated thumbnails ----
    if out_dir and makes:
        os.makedirs(out_dir, exist_ok=True)
        want = {m["frame"]: k for k, m in enumerate(makes)}   # frame idx -> make index
        cap2 = cv2.VideoCapture(source)
        j = 0
        while want:
            ret, frame = cap2.read()
            if not ret:
                break
            if j in want:
                k = want.pop(j)
                m = makes[k]
                img = annotate(frame, m["ball"], m["rim"], m["zone"], m["time"], W)
                path = os.path.join(out_dir, f"make_{k + 1:03d}_t{m['time']:.2f}s.jpg")
                cv2.imwrite(path, img)
                m["image"] = path
            j += 1
        cap2.release()

    if verbose:
        print(f"Detected {len(makes)} make(s) over {n} frames.")
    return makes, fps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.12)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "output", "simple_makes"))
    ap.add_argument("--cooldown", type=float, default=1.5, dest="cooldown_s")
    ap.add_argument("--gap", type=float, default=0.8, dest="gap_s",
                    help="max seconds between the above and below observations")
    ap.add_argument("--min-above", type=int, default=3, dest="min_above",
                    help="min frames the ball must be seen ABOVE the rim first")
    ap.add_argument("--above-factor", type=float, default=1.8, dest="above_factor",
                    help="how far (in rim-widths) the ABOVE lane extends toward "
                         "court-center — the horizontal reach of the L's top bar")
    ap.add_argument("--interp-gap", type=float, default=0.5, dest="interp_gap_s",
                    help="max seconds of missing ball to bridge by interpolation "
                         "(occlusion at the net/players); longer gaps stay empty")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="inference resolution (default: auto — 1280 for >=2000px-wide sources, else 640)")
    ap.add_argument("--rim-source", default="rfdetr", choices=["local", "rfdetr"], dest="rim_source",
                    help="'local' uses the same YOLO weights for ball+rim; 'rfdetr' keeps ball "
                         "detection local but fetches rim boxes from the local RF-DETR checkpoint")
    ap.add_argument("--rim-weights", default=RFDETR_WEIGHTS, dest="rim_weights",
                    help="path to the RF-DETR checkpoint when --rim-source=rfdetr")
    ap.add_argument("--rim-stride", type=int, default=6, dest="rim_stride",
                    help="run the RF-DETR rim model every N frames (holds position in between)")
    ap.add_argument("--target-fps", type=float, default=None, dest="target_fps",
                    help="decimate analysis to roughly this rate (e.g. 24) for faster "
                         "processing of high-fps (60fps) source video; default analyzes every frame")
    args = ap.parse_args()
    detect(args.source, weights=args.weights, conf=args.conf, device=args.device,
           out_dir=args.out, cooldown_s=args.cooldown_s, gap_s=args.gap_s,
           min_above=args.min_above, above_factor=args.above_factor,
           interp_gap_s=args.interp_gap_s, imgsz=args.imgsz,
           rim_source=args.rim_source, rim_weights=args.rim_weights, rim_stride=args.rim_stride,
           target_fps=args.target_fps)
