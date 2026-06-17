"""
run_batch.py
------------
Run the full pipeline (bag → groundtruth, video → estimates, compare) on all
recordings in rv_snimak_sa_begom and print a summary table.
"""
import subprocess
import sys
from pathlib import Path

TOOLKIT  = Path(__file__).parent
BASE_DIR = Path("/home/adi/Desktop/rv_snimak_sa_begom")
BATCH_OUT = TOOLKIT / "output_batch"

RECORDINGS = [
    ("prvi_video",  "uav_landing_rgb_20260612_141408.avi", "rosbag2_2026_06_12-14_14_07"),
    ("drugi_video", "uav_landing_rgb_20260612_141635.avi", "rosbag2_2026_06_12-14_16_35"),
    ("treci",       "uav_landing_rgb_20260612_141805.avi", "rosbag2_2026_06_12-14_18_05"),
    ("cetvrti",     "uav_landing_rgb_20260612_141913.avi", "rosbag2_2026_06_12-14_19_13"),
    ("peti",        "uav_landing_rgb_20260612_142033.avi", "rosbag2_2026_06_12-14_20_33"),
    ("sesti",       "uav_landing_rgb_20260612_142159.avi", "rosbag2_2026_06_12-14_21_59"),
]

results = {}

for name, video_file, bag_dir in RECORDINGS:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    out_dir    = BATCH_OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = BASE_DIR / name / video_file
    bag_path   = BASE_DIR / name / bag_dir

    # Step 1: bag → groundtruth.npz
    print(f"\n[1/3] Parsing bag...")
    r = subprocess.run(
        [sys.executable, "01_parse_and_export.py",
         "--bag", str(bag_path), "--out", str(out_dir)],
        cwd=TOOLKIT,
    )
    if r.returncode != 0:
        print(f"[!] FAILED: parse_and_export for {name}")
        results[name] = {"error": "parse failed"}
        continue

    # Step 2: video → aruco_estimates.csv
    print(f"\n[2/3] Running DLT detector...")
    r = subprocess.run(
        [sys.executable, "evaluate_dlt.py",
         "--video", str(video_path),
         "--out-dir", str(out_dir),
         "--no-display"],
        cwd=TOOLKIT,
    )
    if r.returncode != 0:
        print(f"[!] FAILED: evaluate_dlt for {name}")
        results[name] = {"error": "dlt failed"}
        continue

    # Step 3: compare → error stats
    print(f"\n[3/3] Comparing estimates vs ground truth...")
    r = subprocess.run(
        [sys.executable, "03_compare_aruco.py",
         "--gt",    str(out_dir / "groundtruth.npz"),
         "--aruco", str(out_dir / "aruco_estimates.csv"),
         "--align-time", "--remove-bias"],
        cwd=TOOLKIT,
        capture_output=True, text=True,
    )
    print(r.stdout)
    if r.stderr.strip():
        print(r.stderr)
    results[name] = r.stdout


# ── Summary ─────────────────────────────────────────────────────────────────
print("\n\n" + "="*70)
print("BATCH SUMMARY")
print("="*70)
for name, output in results.items():
    print(f"\n── {name} ──")
    if isinstance(output, dict):
        print(f"  ERROR: {output['error']}")
        continue
    for line in output.splitlines():
        if any(kw in line for kw in ("Error report", "translation:", "rotation   :")):
            print(" ", line.strip())
