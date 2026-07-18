"""
run_analysis.py — run the 2025 Basketball Video Analysis pipeline on a single clip.

This wraps last year's main.py as a callable subprocess entry point for the Huge
Highlights UI. Given one made-shot clip, it produces a fully annotated tactical
breakdown: player + ball tracking, team assignment, ball possession, passes /
interceptions, a top-down tactical map, and per-player speed & distance.

It is meant to be launched as its own process using this folder's isolated venv
(analysis2025/.venv), so the 2025 project's pinned dependencies never collide
with the current project's. All imports and relative asset paths ("./images",
"models/…") resolve because we chdir into this folder first.

Usage:
    analysis2025/.venv/bin/python analysis2025/run_analysis.py <input_clip> <output_mp4>
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)            # so "./images/..." and "models/..." resolve
sys.path.insert(0, HERE)  # so top-level packages (utils, trackers, ...) import

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from utils import read_video
from trackers import PlayerTracker, BallTracker
from team_assigner import TeamAssigner
from court_keypoint_detector import CourtKeypointDetector
from ball_aquisition import BallAquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from tactical_view_converter import TacticalViewConverter
from speed_and_distance_calculator import SpeedAndDistanceCalculator
from drawers import (
    PlayerTracksDrawer, BallTracksDrawer, CourtKeypointDrawer,
    TeamBallControlDrawer, FrameNumberDrawer, PassInterceptionDrawer,
    TacticalViewDrawer, SpeedAndDistanceDrawer,
)
from configs import (
    PLAYER_DETECTOR_PATH, BALL_DETECTOR_PATH, COURT_KEYPOINT_DETECTOR_PATH,
)


def _save_mp4(frames, out_path, fps=24):
    """Write frames to a browser-playable H.264 mp4 via the bundled ffmpeg."""
    import imageio_ffmpeg, numpy as np, cv2
    h, w = frames[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(
        out_path, (w, h), fps=fps, macro_block_size=None,
        codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
        output_params=["-preset", "veryfast"],
    )
    writer.send(None)
    for f in frames:
        writer.send(cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.uint8).tobytes())
    writer.close()


def analyze(input_video, output_mp4):
    frames = read_video(input_video)
    if not frames:
        raise SystemExit(f"No frames read from {input_video}")

    # per-clip stub cache so repeated analysis of the same clip is fast and
    # different clips never contaminate each other
    stub_dir = tempfile.mkdtemp(prefix="a25_stubs_")

    def stub(name):
        return os.path.join(stub_dir, name)

    player_tracker = PlayerTracker(PLAYER_DETECTOR_PATH)
    ball_tracker = BallTracker(BALL_DETECTOR_PATH)
    court_kp = CourtKeypointDetector(COURT_KEYPOINT_DETECTOR_PATH)

    player_tracks = player_tracker.get_object_tracks(frames, read_from_stub=True, stub_path=stub("player.pkl"))
    ball_tracks = ball_tracker.get_object_tracks(frames, read_from_stub=True, stub_path=stub("ball.pkl"))
    court_kps = court_kp.get_court_keypoints(frames, read_from_stub=True, stub_path=stub("court.pkl"))

    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(
        frames, player_tracks, read_from_stub=True, stub_path=stub("teams.pkl"))

    ball_aq = BallAquisitionDetector().detect_ball_possession(player_tracks, ball_tracks)

    pi = PassAndInterceptionDetector()
    passes = pi.detect_passes(ball_aq, player_assignment)
    interceptions = pi.detect_interceptions(ball_aq, player_assignment)

    tv = TacticalViewConverter(court_image_path="./images/basketball_court.png")
    court_kps = tv.validate_keypoints(court_kps)
    tac_positions = tv.transform_players_to_tactical_view(court_kps, player_tracks)

    sd = SpeedAndDistanceCalculator(tv.width, tv.height, tv.actual_width_in_meters, tv.actual_height_in_meters)
    dist = sd.calculate_distance(tac_positions)
    speed = sd.calculate_speed(dist)

    out = PlayerTracksDrawer().draw(frames, player_tracks, player_assignment, ball_aq)
    out = BallTracksDrawer().draw(out, ball_tracks)
    out = CourtKeypointDrawer().draw(out, court_kps)
    out = FrameNumberDrawer().draw(out)
    out = TeamBallControlDrawer().draw(out, player_assignment, ball_aq)
    out = PassInterceptionDrawer().draw(out, passes, interceptions)
    out = SpeedAndDistanceDrawer().draw(out, player_tracks, dist, speed)
    out = TacticalViewDrawer().draw(out, tv.court_image_path, tv.width, tv.height,
                                    tv.key_points, tac_positions, player_assignment, ball_aq)

    os.makedirs(os.path.dirname(os.path.abspath(output_mp4)), exist_ok=True)
    _save_mp4(out, output_mp4)
    return output_mp4


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: run_analysis.py <input_clip> <output_mp4>")
    result = analyze(sys.argv[1], os.path.abspath(sys.argv[2]))
    print(f"ANALYSIS_OK {result}")
