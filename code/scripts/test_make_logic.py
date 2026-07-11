"""
test_make_logic.py — unit-test the make/miss geometry in isolation.

Feeds MakeDetector synthetic (ball_box, rim_box) sequences with KNOWN answers,
so we verify the decision logic works independently of detector quality.

Rim box used: [500, 200, 600, 240]  (left=500, top=200, right=600, bottom=240)
  -> x-gate (margin 0.5): 450..650
Ball boxes are built around a center (x, y) with a fixed radius (BALL_R),
roughly matching the scale of the rim box, so the box-contact test (ball's
top edge vs. rim's bottom edge) has something realistic to work with.
Run:  python scripts/test_make_logic.py
"""
from make_detector import MakeDetector, dedup_makes

RIM = [500, 200, 600, 240]
FPS = 30
BALL_R = 20  # synthetic ball "radius" in px (diameter ~ rim height)


def box(cx, cy, r=BALL_R):
    return (cx - r, cy - r, cx + r, cy + r)


def run(frames):
    """frames: list of (ball_box_or_None). Rim fed every frame. Returns #makes."""
    d = MakeDetector(fps=FPS)
    for i, ball in enumerate(frames):
        d.update(i, ball, RIM)
    return len(d.makes)


def col(x, ys):
    """Helper: ball at fixed x, descending through the given center-y values."""
    return [box(x, y) for y in ys]


SCENARIOS = [
    # name, frames, expected makes
    ("Clean make (straight down through rim)",
     col(550, [150, 190, 230, 260, 290]), 1),

    ("Wide miss (ball passes outside the rim x-range)",
     col(300, [150, 200, 250, 300]), 0),

    ("Short shot (ball never reaches the rim, stays above)",
     col(550, [150, 170, 185, 195, 185, 170]), 0),

    ("Rim-out bounce (dips to rim, then bounces back up)",
     col(550, [150, 200, 235, 238, 210, 170, 140]), 0),

    ("Occlusion gap (ball above, lost for 5 frames, reappears below)",
     [box(550, 150)] + [None] * 5 + [box(550, 260), box(550, 300)], 1),

    ("No cooldown double-count (one make, extra below-samples right after)",
     col(550, [150, 260, 265, 270, 262, 268]), 1),

    ("Fast ball (one frame above, next frame already below)",
     [box(550, 140), box(550, 300)], 1),

    ("Upward pass (rebound rising through rim — NOT a make)",
     col(550, [300, 260, 230, 190, 140]), 0),

    ("Edge-of-rim make (ball at right edge, still within margin)",
     col(645, [150, 200, 260, 300]), 1),

    ("Just outside the rim margin (should NOT count)",
     col(670, [150, 200, 260, 300]), 0),
]


def main():
    passed = 0
    print("=" * 62)
    for name, frames, expected in SCENARIOS:
        got = run(frames)
        ok = got == expected
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"        expected {expected} make(s), got {got}")
    print("=" * 62)

    # Cooldown + second real make: make, wait past cooldown (45f), make again -> 2
    d = MakeDetector(fps=FPS)
    seq = col(550, [150, 190, 260])              # make #1 around frame 2
    seq += [box(550, 300)] * 50                   # sit below through the cooldown
    seq += col(550, [150, 190, 260])              # a fresh make after cooldown
    for i, ball in enumerate(seq):
        d.update(i, ball, RIM)
    ok = len(d.makes) == 2
    passed += ok
    print(f"[{'PASS' if ok else 'FAIL'}] Cooldown then a second real make")
    print(f"        expected 2 make(s), got {len(d.makes)}")
    print("=" * 62)

    # Stale rim guard: rim seen once, then camera pans away (rim gone 70 frames),
    # ball later drops through the OLD rim spot -> must NOT count (rim unreliable).
    d = MakeDetector(fps=FPS)
    d.update(0, box(550, 150), RIM)              # rim seen, ball above
    for i in range(1, 70):
        d.update(i, None, None)                  # rim gone (panned away), no ball
    d.update(70, box(550, 150), None)             # ball above old spot, rim still unseen
    d.update(71, box(550, 300), None)             # ball below old spot — but rim is stale
    ok = len(d.makes) == 0
    passed += ok
    print(f"[{'PASS' if ok else 'FAIL'}] Stale rim ignored (camera panned away)")
    print(f"        expected 0 make(s), got {len(d.makes)}")
    print("=" * 62)

    # De-dup guard: two makes within 7s -> earlier canceled, later kept.
    # (Mirrors the real TestClip case: real make 17.35s + false pos 21.18s.)
    dd_cases = [
        ("two makes 3.8s apart -> keep later",
         [{"time": 17.35, "frame": 1041}, {"time": 21.18, "frame": 1271}],
         [21.18]),
        ("two makes 12s apart -> keep both",
         [{"time": 5.30, "frame": 318}, {"time": 17.35, "frame": 1041}],
         [5.30, 17.35]),
        ("full TestClip set (5.3, 17.35, 21.18) -> 5.3 + 21.18",
         [{"time": 5.30, "frame": 318}, {"time": 17.35, "frame": 1041},
          {"time": 21.18, "frame": 1271}],
         [5.30, 21.18]),
        ("empty -> empty", [], []),
    ]
    for name, inp, expected_times in dd_cases:
        got = [round(m["time"], 2) for m in dedup_makes(inp, min_gap_s=7.0)]
        ok = got == expected_times
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] dedup: {name}")
        print(f"        expected {expected_times}, got {got}")
    print("=" * 62)

    total = len(SCENARIOS) + 2 + len(dd_cases)
    print(f"{passed}/{total} scenarios passed.")
    return passed == total


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
