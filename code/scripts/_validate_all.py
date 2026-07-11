"""
_validate_all.py — run the REAL detect() pipeline on every test clip and score
makes against expected timestamps. Confirms a change helps the target clip without
regressing the others. Debug/validation helper, not shipped.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from simple_make import detect
from make_detector import dedup_makes

TC = os.path.expanduser("~/Desktop/TestClips")
CLIPS = {
    "Testing": ("/Users/hugonelson/Desktop/Basketball Analysis/Testing.mp4", [12, 45]),
    "lal_vs_gs": (f"{TC}/broadcast_lal_vs_gs_2makes.mp4", [5.3, 17.35]),
    "gs_vs_mem": (f"{TC}/broadcast_gs_vs_mem_1make.mp4", [5.5]),
    "utah_jazz": (f"{TC}/broadcast_utah_jazz_1make.mp4", [8.45]),
    "wizards": (f"{TC}/broadcast_wizards_vs_warriors_1make.mp4", [5.8]),
    "highschool": (f"{TC}/highschool_riverdale_falcons_2make.mp4", [3.58]),
    "carom (must be 0)": (f"{TC}/edgecase_carom_false_positive.mp4", []),
    "flat_angle": (f"{TC}/edgecase_flat_angle_missed_make.mp4", [1]),
}


def score(exp, makes):
    matched = 0; used = set()
    for e in exp:
        for j, m in enumerate(makes):
            if j not in used and abs(m - e) <= 3.0:
                matched += 1; used.add(j); break
    return matched, len(exp) - matched, len(makes) - matched


if __name__ == "__main__":
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    results = {}
    for name, (path, exp) in CLIPS.items():
        if only and not any(o in name for o in only):
            continue
        if not os.path.exists(path):
            print(f"{name}: MISSING {path}"); continue
        makes, fps = detect(path, rim_source="rfdetr", rim_stride=6, verbose=False)
        makes = dedup_makes(makes, min_gap_s=7.0)
        times = [round(m["time"], 1) for m in makes]
        m, ms, fp = score(exp, times)
        results[name] = (times, m, ms, fp)
        carom_flag = "  <<< CAROM FP" if ("carom" in name and len(times) > 0) else ""
        print(f"{name:22s} makes={str(times):22s} match={m} miss={ms} fp={fp}{carom_flag}", flush=True)
    tm = sum(r[1] for r in results.values())
    tms = sum(r[2] for r in results.values())
    tfp = sum(r[3] for r in results.values())
    print(f"\nTOTAL matched={tm} missed={tms} false_pos={tfp}")
