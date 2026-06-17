"""
03_compare_aruco.py
-------------------
Compare your ArUco pose estimates against the mocap ground truth.

EXPECTED INPUT  (output/aruco_estimates.csv) -- one row per detection:
    t            time in seconds on the SAME clock you want to align on
                 (or provide t_ns; see --time-col). If your clock differs
                 from the bag's, use estimate_time_offset().
    x, y, z      marker translation in the camera optical frame [m]  (= tvec)
    orientation, EITHER:
        qx, qy, qz, qw     quaternion, OR
        rx, ry, rz         Rodrigues rotation vector (= cv2 rvec)

The script:
  * loads ground truth (output/groundtruth.npz),
  * time-aligns the estimates to the GT clock (nearest / interpolation),
  * optionally removes a constant SE(3) bias (mocap-rigidbody vs optical-frame
    calibration -- see README "Calibration caveat"),
  * reports translation & rotation error stats and writes fig_error.png.

Run with no aruco file present -> it synthesises a noisy example so you can
see the whole pipeline working, then replace it with your real data.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import mocap_rv as mc

OUT = Path("output")
_out_dir = OUT


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #
def load_groundtruth(gt_path: Path = None):
    path = gt_path if gt_path is not None else (OUT / "groundtruth.npz")
    d = np.load(path)
    frozen = d["frozen"].tolist() if "frozen" in d.files else []
    return d["t"], d["rel_pos"], d["rel_quat"], frozen


def load_aruco(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t, pos (N,3), quat (N,4) xyzw) from the user's csv."""
    df = pd.read_csv(path)
    t = df["t"].to_numpy() if "t" in df else df["t_ns"].to_numpy() / 1e9
    pos = df[["x", "y", "z"]].to_numpy()
    if {"qx", "qy", "qz", "qw"}.issubset(df.columns):
        quat = df[["qx", "qy", "qz", "qw"]].to_numpy()
    elif {"rx", "ry", "rz"}.issubset(df.columns):
        quat = R.from_rotvec(df[["rx", "ry", "rz"]].to_numpy()).as_quat()
    else:
        raise ValueError("aruco csv needs either qx..qw or rx,ry,rz columns")
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return t, pos, quat


# --------------------------------------------------------------------------- #
#  Alignment helpers
# --------------------------------------------------------------------------- #
def estimate_time_offset(t_gt, dist_gt, t_est, dist_est, max_off=5.0, step=0.005):
    """Brute-force time offset (seconds) that best aligns two distance signals.

    Returns the offset to ADD to t_est so it lines up with t_gt. Useful when
    the ArUco log and the mocap bag ran on different clocks.
    """
    offs = np.arange(-max_off, max_off + step, step)
    best, best_off = np.inf, 0.0
    for o in offs:
        di = np.interp(t_gt, t_est + o, dist_est,
                       left=np.nan, right=np.nan)
        m = np.isfinite(di)
        if m.sum() < 10:
            continue
        err = np.sqrt(np.mean((di[m] - dist_gt[m]) ** 2))
        if err < best:
            best, best_off = err, o
    return best_off


def align_gt_to_est(t_gt, pos_gt, quat_gt, t_est):
    """Interpolate GT onto the estimate timestamps (SLERP for rotation)."""
    from scipy.spatial.transform import Slerp
    t_clip = np.clip(t_est, t_gt[0], t_gt[-1])
    pos = np.column_stack([np.interp(t_clip, t_gt, pos_gt[:, i]) for i in range(3)])
    quat = Slerp(t_gt, R.from_quat(quat_gt))(t_clip).as_quat()
    return pos, quat


def estimate_constant_offset(pos_gt, quat_gt, pos_est, quat_est):
    """Best-fit constant rotation + translation removing systematic bias.

    Models  p_est ~= Rb @ p_gt + tb   and a constant rotation offset Qb such
    that  R_est ~= Qb @ R_gt.  This isolates random tracking error from the
    fixed (and unknown) calibration between the mocap rigid-body frames and the
    true optical / marker frames.  Returns corrected GT (pos, quat).
    """
    # translation: Umeyama (rotation+translation, no scale)
    mu_g, mu_e = pos_gt.mean(0), pos_est.mean(0)
    Pg, Pe = pos_gt - mu_g, pos_est - mu_e
    U, _, Vt = np.linalg.svd(Pe.T @ Pg)
    d = np.sign(np.linalg.det(U @ Vt))
    Rb = U @ np.diag([1, 1, d]) @ Vt
    tb = mu_e - Rb @ mu_g
    pos_corr = (Rb @ pos_gt.T).T + tb
    # rotation: mean offset Qb = mean(R_est * R_gt^-1)
    Roff = R.from_quat(quat_est) * R.from_quat(quat_gt).inv()
    Qb = Roff.mean()
    quat_corr = (Qb * R.from_quat(quat_gt)).as_quat()
    return pos_corr, quat_corr


# --------------------------------------------------------------------------- #
#  Metrics + report
# --------------------------------------------------------------------------- #
def report(t, pos_gt, quat_gt, pos_est, quat_est, tag=""):
    terr = np.linalg.norm(pos_est - pos_gt, axis=1)
    rerr = mc.angular_distance_deg(quat_est, quat_gt)

    def stats(a, u):
        return (f"mean={a.mean():.4f}{u}  median={np.median(a):.4f}{u}  "
                f"rmse={np.sqrt(np.mean(a**2)):.4f}{u}  "
                f"p95={np.percentile(a,95):.4f}{u}  max={a.max():.4f}{u}")

    print(f"\n=== Error report {tag} ({len(t)} matched samples) ===")
    print("  translation:", stats(terr, " m"))
    print("  rotation   :", stats(rerr, " deg"))

    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axs[0].plot(t, terr, lw=0.7, color="tab:red")
    axs[0].set_ylabel("translation error [m]"); axs[0].grid(alpha=.3)
    axs[0].set_title(f"ArUco vs mocap ground truth {tag}")
    axs[1].plot(t, rerr, lw=0.7, color="tab:purple")
    axs[1].set_ylabel("rotation error [deg]"); axs[1].set_xlabel("time [s]")
    axs[1].grid(alpha=.3)
    fig.tight_layout()
    name = f"fig_error{('_'+tag) if tag else ''}.png".replace(" ", "_")
    fig.savefig(_out_dir / name); plt.close(fig)
    print(f"  wrote {_out_dir/name}")
    return terr, rerr


# --------------------------------------------------------------------------- #
def make_synthetic_example(t_gt, pos_gt, quat_gt):
    """Create a believable noisy ArUco log to demonstrate the pipeline."""
    rng = np.random.default_rng(0)
    # subsample to ~30 Hz (a camera is slower than the 79 Hz mocap)
    idx = np.arange(0, len(t_gt), 3)
    t = t_gt[idx] + 0.7                       # pretend clocks differ by 0.7 s
    # constant calibration bias + gaussian noise (grows a bit with distance)
    bias_R = R.from_euler("xyz", [2, -3, 1], degrees=True)
    bias_t = np.array([0.01, -0.02, 0.03])
    dist = np.linalg.norm(pos_gt[idx], axis=1, keepdims=True)
    pos = (bias_R.apply(pos_gt[idx]) + bias_t
           + rng.normal(0, 0.01, size=(len(idx), 3)) * (1 + dist / 4))
    noise_R = R.from_rotvec(rng.normal(0, np.radians(1.5), size=(len(idx), 3)))
    quat = (noise_R * bias_R * R.from_quat(quat_gt[idx])).as_quat()
    df = pd.DataFrame({"t": t, "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
                       "qx": quat[:, 0], "qy": quat[:, 1],
                       "qz": quat[:, 2], "qw": quat[:, 3]})
    p = OUT / "aruco_estimates_example.csv"
    df.to_csv(p, index=False)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aruco", type=Path, default=OUT / "aruco_estimates.csv",
                    help="your ArUco estimates csv")
    ap.add_argument("--gt", type=Path, default=None,
                    help="groundtruth.npz path (default: output/groundtruth.npz)")
    ap.add_argument("--align-time", action="store_true",
                    help="auto-estimate a constant time offset between clocks")
    ap.add_argument("--remove-bias", action="store_true",
                    help="remove constant SE(3) calibration offset before scoring")
    args = ap.parse_args()

    global _out_dir
    _out_dir = args.gt.parent if args.gt is not None else OUT
    _out_dir.mkdir(parents=True, exist_ok=True)

    t_gt, pos_gt, quat_gt, frozen = load_groundtruth(args.gt)
    dist_gt = np.linalg.norm(pos_gt, axis=1)

    aruco_path = args.aruco
    if not aruco_path.exists():
        print(f"[i] {aruco_path} not found -> generating a synthetic example.")
        aruco_path = make_synthetic_example(t_gt, pos_gt, quat_gt)
        args.align_time = True  # the synthetic log has a 0.7 s clock offset

    t_est, pos_est, quat_est = load_aruco(aruco_path)

    if args.align_time:
        dist_est = np.linalg.norm(pos_est, axis=1)
        off = estimate_time_offset(t_gt, dist_gt, t_est, dist_est)
        print(f"[i] estimated time offset: {off:+.3f} s (added to estimate clock)")
        t_est = t_est + off

    # keep estimates that fall inside the GT time span
    m = (t_est >= t_gt[0]) & (t_est <= t_gt[-1])
    t_est, pos_est, quat_est = t_est[m], pos_est[m], quat_est[m]

    # drop estimates that fall in frozen (occluded) ground-truth windows
    if frozen:
        keep = mc.valid_time_mask(t_est, frozen)
        print(f"[i] excluding {(~keep).sum()} samples in {len(frozen)} frozen "
              f"GT windows")
        t_est, pos_est, quat_est = t_est[keep], pos_est[keep], quat_est[keep]

    if len(t_est) == 0:
        print("[!] No valid samples remain after filtering — cannot compute errors.")
        print("    (All estimates fall in frozen GT windows or outside the GT time span.)")
        return

    g_pos, g_quat = align_gt_to_est(t_gt, pos_gt, quat_gt, t_est)

    report(t_est, g_pos, g_quat, pos_est, quat_est, tag="raw")

    if args.remove_bias:
        g_pos2, g_quat2 = estimate_constant_offset(g_pos, g_quat, pos_est, quat_est)
        report(t_est, g_pos2, g_quat2, pos_est, quat_est, tag="bias-removed")


if __name__ == "__main__":
    main()
