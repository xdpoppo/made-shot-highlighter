"""
make_detector.py — Phase 5: the core make/miss geometry (Level 1).

Given the ball's tracked bounding box and the rim box each frame, decide when
a shot went in: the ball must travel from ABOVE the rim to a frame where it has
physically passed through the rim plane, staying within the rim's horizontal
range, moving downward. Includes the three fixes:

  1. In-front illusion  -> require actual contact: the ball's own TOP edge must
                           reach the rim's BOTTOM edge (not just its center
                           crossing an arbitrary line — this uses the ball's
                           real measured size instead of a fudge factor).
  2. Net/player occlusion -> bridge short detection gaps (above and below samples
                             may be several frames apart even if the ball vanishes).
  3. Rim-out bounce      -> require net downward motion between the above/below samples.

Plus a cooldown so one make isn't counted twice.

This module is pure logic (MakeDetector) + a convenience runner (process_video)
that drives Ultralytics tracking and returns the detected make events.
"""
import os
from collections import deque

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class MakeDetector:
    def __init__(self, fps, x_margin=0.5, gap_frames=15, cooldown_s=1.5,
                 rim_alpha=0.15, rim_max_stale_s=2.0,
                 static_eps=3.0, static_window_s=0.15, close_frac=0.4):
        """
        x_margin:   how far outside the rim box (as a fraction of rim width) the
                    ball may be and still count as "over the hoop".
        gap_frames: max frames between the 'above' and 'below' observations
                    (bridges occlusion).
        cooldown_s: seconds to wait after a make before allowing another.
        rim_alpha:  EMA smoothing for the (roughly stationary) rim band.
        rim_max_stale_s: if the rim hasn't actually been detected within this many
                    seconds, don't evaluate makes (its remembered position is
                    unreliable — e.g. the camera panned away).
        static_eps/static_window_s: a real ball in live play is essentially
                    never pixel-frozen for static_window_s straight — that's
                    the signature of background clutter near the hoop (a
                    cooler, bench gear, equipment) getting misclassified as
                    the ball. Positions that haven't moved more than static_eps
                    px over that window are dropped as noise before they can
                    masquerade as an "above the rim" sample.
        close_frac: detection boxes are noisiest exactly when the ball is
                    dropping through the net (partial occlusion shrinks/shifts
                    the box), so we don't require the ball's top edge to have
                    fully passed the rim's bottom edge — just to have closed
                    to within close_frac of the ball's OWN measured height.
                    Scaled to the ball's size (not a flat pixel count or a
                    fraction of the rim) so it stays meaningful whether the
                    camera is close-up or far away.
        """
        self.fps = fps
        self.x_margin = x_margin
        self.gap_frames = gap_frames
        self.cooldown_frames = cooldown_s * fps
        self.rim_alpha = rim_alpha
        self.rim_max_stale = rim_max_stale_s * fps
        self.static_eps = static_eps
        self.static_window = max(3, round(static_window_s * fps))
        self.close_frac = close_frac
        self.ban_radius = static_eps * 3
        self.banned_zones = []               # confirmed clutter spots: [(cx, cy), ...]

        self.rim = None                      # dict: left, top, right, bottom
        self.ball_hist = deque(maxlen=90)    # (frame_idx, cx, cy)
        self.last_make_frame = -1e9
        self.last_rim_frame = -1e9           # last frame the rim was actually seen
        self.makes = []                      # list of {frame, time}

    def _is_banned(self, cx, cy):
        """True if (cx, cy) falls in a screen location already confirmed to be
        static clutter (a cooler, bench gear, etc.) rather than the ball."""
        return any(abs(cx - bx) <= self.ban_radius and abs(cy - by) <= self.ban_radius
                  for bx, by in self.banned_zones)

    def _is_static_noise(self, cx, cy):
        """True if the last static_window ball positions (incl. this one) are
        all clustered within static_eps px — almost certainly a stationary
        background object, not a real ball in play."""
        if len(self.ball_hist) < self.static_window - 1:
            return False
        recent = list(self.ball_hist)[-(self.static_window - 1):]
        xs = [p[1] for p in recent] + [cx]
        ys = [p[2] for p in recent] + [cy]
        return (max(xs) - min(xs) <= self.static_eps) and (max(ys) - min(ys) <= self.static_eps)

    def _near_rim(self, cx, cy):
        """True if (cx, cy) falls within (or just around) the current rim
        band. A real ball can legitimately sit almost still here for a
        while — rattling on the front rim before dropping through — so
        stillness at this specific location isn't good evidence of static
        background clutter the way stillness elsewhere on screen is."""
        if self.rim is None:
            return False
        rim = self.rim
        rw = rim["right"] - rim["left"]
        rh = max(1.0, rim["bottom"] - rim["top"])
        # Wider than the strict make-contact x-gate on purpose: this only
        # decides whether to exempt a position from the clutter filter, not
        # whether a make occurred, so it can afford to be generous — a ball
        # approaching the rim (e.g. from the side, just outside the tight
        # contact margin) can still legitimately settle/rattle here.
        margin = 2 * self.x_margin
        xlo, xhi = rim["left"] - margin * rw, rim["right"] + margin * rw
        ylo, yhi = rim["top"] - rh, rim["bottom"] + rh
        return xlo <= cx <= xhi and ylo <= cy <= yhi

    def _update_rim(self, box):
        if box is None:
            return
        l, t, r, b = box
        if self.rim is None:
            self.rim = dict(left=l, top=t, right=r, bottom=b)
        else:
            a = self.rim_alpha
            self.rim["left"] = (1 - a) * self.rim["left"] + a * l
            self.rim["top"] = (1 - a) * self.rim["top"] + a * t
            self.rim["right"] = (1 - a) * self.rim["right"] + a * r
            self.rim["bottom"] = (1 - a) * self.rim["bottom"] + a * b

    def update(self, frame_idx, ball_box, rim_box):
        """Feed one frame. ball_box is (x1, y1, x2, y2) or None.
        Returns frame_idx if a make is registered, else None."""
        if rim_box is not None:
            self._update_rim(rim_box)
            self.last_rim_frame = frame_idx

        ball_center = None
        if ball_box is not None:
            bx1, by1, bx2, by2 = ball_box
            cx0, cy0 = (bx1 + bx2) / 2, (by1 + by2) / 2
            if self._is_banned(cx0, cy0):
                ball_box = None
            elif self._is_static_noise(cx0, cy0) and not self._near_rim(cx0, cy0):
                # Confirmed clutter, not a real ball — and the points we already
                # recorded on our way to this conclusion were clutter too, so
                # purge them retroactively (otherwise a stale "above the rim"
                # sample can linger in ball_hist and combine with a later, real
                # below-rim sample into a false make). Also blacklist this
                # screen location so the same object can't re-latch on later
                # once enough fresh history has scrolled past to hide it again.
                while self.ball_hist:
                    _, hx, hy = self.ball_hist[-1]
                    if abs(hx - cx0) <= self.static_eps and abs(hy - cy0) <= self.static_eps:
                        self.ball_hist.pop()
                    else:
                        break
                self.banned_zones.append((cx0, cy0))
                ball_box = None
            else:
                self.ball_hist.append((frame_idx, cx0, cy0))
                ball_center = (cx0, cy0)

        if self.rim is None or ball_box is None:
            return None
        # Don't trust a stale rim position (camera may have panned away).
        if frame_idx - self.last_rim_frame > self.rim_max_stale:
            return None
        if frame_idx - self.last_make_frame < self.cooldown_frames:
            return None

        rim = self.rim
        rw = rim["right"] - rim["left"]
        xlo = rim["left"] - self.x_margin * rw
        xhi = rim["right"] + self.x_margin * rw

        bx1, by1, bx2, by2 = ball_box
        cx, cy = ball_center
        ball_h = max(1.0, by2 - by1)
        # The ball has (near-)physically passed through the rim plane once its
        # own TOP edge reaches the rim's BOTTOM edge — using the ball's real
        # measured size instead of an arbitrary fraction of the rim's height
        # (fix #1). Boxes are noisiest exactly here (the net partially hides
        # the ball as it drops), so allow the top edge to be within close_frac
        # of the ball's own height of the rim's bottom, not just past it.
        if by1 < rim["bottom"] - self.close_frac * ball_h or not (xlo <= cx <= xhi):
            return None

        # Look back for an ABOVE-rim sample within the gap window, over the hoop,
        # with downward motion between then and now (fixes #2 and #3).
        for (f0, x0, y0) in reversed(self.ball_hist):
            if frame_idx - f0 > self.gap_frames:
                break
            if y0 < rim["top"] and (xlo <= x0 <= xhi) and (cy - y0) > 0:
                self.last_make_frame = frame_idx
                self.makes.append({"frame": frame_idx, "time": frame_idx / self.fps})
                return frame_idx
        return None


def dedup_makes(makes, min_gap_s=7.0):
    """Collapse makes that fire too close together in time.

    You can't legitimately score twice on the same possession within a few
    seconds, so when two makes land within min_gap_s of each other one of them
    is a miscount (e.g. a stray misdetection near the rim). We cancel the
    EARLIER of the pair and keep the later one.

    This is safe even when the later timestamp is the false one: the highlight
    clip cut around a make reaches ~10s BACK, and the two makes are <min_gap_s
    apart, so the surviving (later) make's clip still contains the real make's
    moment regardless of which timestamp was the true one.

    `makes` must be in ascending time order (process_video emits them that way).
    Returns a new list; the input is not modified.
    """
    kept = []
    for i, m in enumerate(makes):
        nxt = makes[i + 1] if i + 1 < len(makes) else None
        if nxt is not None and (nxt["time"] - m["time"]) < min_gap_s:
            continue  # a later make is close behind -> this one is the miscount
        kept.append(m)
    return kept


def _pick_centers(boxes, names, prev_ball=None, max_jump=120, ball_hist=None):
    """From one frame's detections, return (ball_box_xyxy, rim_box_xyxy).

    Static objects near the hoop (coolers, equipment, a referee's whistle) can
    outscore the real ball on confidence, especially during fast/blurry drives.
    When we have a recent ball position, only accept a candidate within
    max_jump px of it (the ball can't teleport frame-to-frame) — the caller
    should grow max_jump with how many frames it's been since the last real
    detection, so genuine reacquisition after occlusion still works. If
    nothing is within range, report no detection this frame rather than
    grabbing an implausible high-confidence outlier. Pure confidence is only
    used when there's no prior position at all (cold start).

    ball_hist: deque of (frame_idx, cx, cy). When the ball has clearly been
    moving fast (e.g. mid-arc), the flat gap-scaled max_jump can be too tight
    and cause the tracker to lose the real ball to a nearer, wrong candidate —
    so recent measured speed WIDENS (never tightens) the acceptance radius.
    """
    ball_id = next((i for i, n in names.items() if n.lower() == "ball"), None)
    rim_id = next((i for i, n in names.items() if n.lower() in ("rim", "hoop")), None)
    ball_candidates = []
    best_rim = None
    best_rim_c = -1.0
    for b in boxes:
        cls = int(b.cls)
        conf = float(b.conf)
        xyxy = [float(v) for v in b.xyxy[0]]
        if cls == ball_id:
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            ball_candidates.append((conf, cx, cy, xyxy))
        elif cls == rim_id and conf > best_rim_c:
            best_rim_c = conf
            best_rim = xyxy

    best_ball = None
    if ball_candidates:
        if prev_ball is not None:
            px, py = prev_ball

            max_speed = 120.0
            if ball_hist and len(ball_hist) >= 3:
                recent = list(ball_hist)[-3:]
                speeds = []
                for i in range(1, len(recent)):
                    _, x0, y0 = recent[i - 1]
                    _, x1, y1 = recent[i]
                    speeds.append(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
                if speeds:
                    max_speed = max(speeds) * 3.0

            nearest = min(ball_candidates,
                         key=lambda c: (c[1] - px) ** 2 + (c[2] - py) ** 2)
            dist = ((nearest[1] - px) ** 2 + (nearest[2] - py) ** 2) ** 0.5
            if dist <= max(max_jump, max_speed * 1.5):
                best_ball = nearest[3]
            # else: nothing plausible nearby -> leave as None, don't guess
        else:
            best_ball = max(ball_candidates, key=lambda c: c[0])[3]
    return best_ball, best_rim


def process_video(weights, source, conf=0.25, device="mps", verbose=True,
                  debug_out=None, vid_stride=1, **det_kwargs):
    """Run tracking over a video and return (makes, fps). Single detection pass.

    vid_stride>1 processes only every Nth frame (~N× faster). We still feed the
    detector the TRUE frame index (kept_idx * vid_stride) so cooldown, rim-staleness
    and the make timestamps stay in real video time — clips stay correctly centered.

    If debug_out is a path, also writes an annotated video (at fps/vid_stride so it
    plays at real speed) showing the rim band, ball trajectory, and a MAKE flash.
    """
    import cv2
    from ultralytics import YOLO

    vid_stride = max(1, int(vid_stride))
    cap0 = cv2.VideoCapture(source)
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    # Keep the occlusion-bridge window wide enough to span a few kept frames even
    # at large strides (kept frames are vid_stride real-frames apart).
    det_kwargs["gap_frames"] = max(int(det_kwargs.get("gap_frames", 15)), 5 * vid_stride)

    model = YOLO(weights)
    detector = MakeDetector(fps=fps, **det_kwargs)

    writer = None
    if debug_out:
        os.makedirs(os.path.dirname(os.path.abspath(debug_out)), exist_ok=True)
        writer = cv2.VideoWriter(debug_out, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps / vid_stride, (W, H))
    flash = 0  # kept-frames remaining to show the MAKE banner

    kept = 0
    # Plain detection, not model.track(): our own continuity logic in
    # _pick_centers replaces ByteTrack's persistence, and track() was found to
    # silently drop the faint (low-confidence, motion-blurred) ball boxes near
    # the rim that this pipeline actually needs.
    for r in model.predict(source=source, stream=True, conf=conf,
                           device=device, vid_stride=vid_stride, verbose=False):
        true_frame = kept * vid_stride  # index in the ORIGINAL video timeline
        hist = detector.ball_hist
        if len(hist) >= 2:
            (last_f, x1, y1), (_, x0, y0) = hist[-1], hist[-2]
            pred_ball = (x1 + (x1 - x0), y1 + (y1 - y0))  # extrapolate through fast motion
        elif hist:
            last_f, x1, y1 = hist[-1]
            pred_ball = (x1, y1)
        else:
            last_f, pred_ball = true_frame, None
        # Allow more slack the longer it's been since a real detection (real
        # occlusion means the ball may have moved further), but keep a tight
        # bound frame-to-frame so we don't accept implausible teleports.
        gap = max(1, true_frame - last_f)
        dyn_max_jump = min(600, 100 + 60 * gap)
        ball_box, rim_box = _pick_centers(r.boxes, model.names, prev_ball=pred_ball, max_jump=dyn_max_jump, ball_hist=hist)
        hit = detector.update(true_frame, ball_box, rim_box)
        if hit is not None:
            if verbose:
                print(f"  MAKE @ frame {hit}  (t={hit/fps:.2f}s)")
            flash = int(fps / vid_stride)  # ~1 second banner (in kept frames)

        if writer is not None:
            flash = _draw_debug(r.orig_img, detector, ball_box, flash, writer)
        kept += 1

    if writer is not None:
        writer.release()
    if verbose:
        print(f"Detected {len(detector.makes)} make(s) over {kept} processed "
              f"frames (stride {vid_stride}).")
        if debug_out:
            print(f"Debug video: {debug_out}")
    return detector.makes, fps


def _draw_debug(frame, detector, ball_box, flash, writer):
    import cv2
    f = frame.copy()
    rim = detector.rim
    if rim is not None:
        rw = rim["right"] - rim["left"]
        xlo = int(rim["left"] - detector.x_margin * rw)
        xhi = int(rim["right"] + detector.x_margin * rw)
        # rim box (red) and the x-range gate (yellow), down to the rim's bottom
        cv2.rectangle(f, (int(rim["left"]), int(rim["top"])),
                      (int(rim["right"]), int(rim["bottom"])), (0, 0, 255), 2)
        cv2.line(f, (xlo, int(rim["top"])), (xlo, int(rim["bottom"])), (0, 220, 220), 1)
        cv2.line(f, (xhi, int(rim["top"])), (xhi, int(rim["bottom"])), (0, 220, 220), 1)
    # ball trajectory (orange dots) + current ball box (the contact test uses
    # this box's own top edge against the rim's bottom edge, not a fixed line)
    for (_, x, y) in list(detector.ball_hist)[-25:]:
        cv2.circle(f, (int(x), int(y)), 3, (0, 140, 255), -1)
    if ball_box is not None:
        x1, y1, x2, y2 = map(int, ball_box)
        cv2.rectangle(f, (x1, y1), (x2, y2), (0, 140, 255), 2)
    if flash > 0:
        cv2.putText(f, "MAKE", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 6)
        flash -= 1
    writer.write(f)
    return flash


if __name__ == "__main__":
    import argparse
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--weights", default=os.path.join(PROJECT_ROOT, "models", "ball_detector_v2025", "weights", "best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--debug", action="store_true", help="write annotated debug video")
    ap.add_argument("--debug-out", default=os.path.join(PROJECT_ROOT, "output", "makes_debug.mp4"))
    ap.add_argument("--vid-stride", type=int, default=1, dest="vid_stride",
                    help="process every Nth frame (N=3 ~3x faster; too high can miss fast makes)")
    args = ap.parse_args()
    makes, fps = process_video(args.weights, args.source, conf=args.conf, device=args.device,
                               debug_out=args.debug_out if args.debug else None,
                               vid_stride=args.vid_stride)
    for m in makes:
        print(f"make at {m['time']:.2f}s (frame {m['frame']})")
