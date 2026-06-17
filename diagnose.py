"""
diagnose.py
-----------
Pinpoints why ArUco-vs-mocap errors are large. Checks, for BOTH possible
camera/marker role assignments:
  * time alignment quality (distance-signal correlation),
  * translation scale (slope of estimated vs true distance; should be ~1),
  * residual after best-fit rigid alignment (reveals the correct roles +
    the true noise floor).

Run:  python diagnose.py
Reads output/aruco_estimates.csv (your data) and the bag. Writes
output/fig_diagnose.png and prints a verdict.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R, Slerp

import mocap_rv as mc

BAG = "rv_mocap/rosbag2_2026_06_12-13_53_18"
OUT = Path("output")


def load_aruco(path):
    df = pd.read_csv(path)
    t = df["t"].to_numpy() if "t" in df else df["t_ns"].to_numpy() / 1e9
    pos = df[["x", "y", "z"]].to_numpy()
    if {"qx", "qy", "qz", "qw"}.issubset(df.columns):
        quat = df[["qx", "qy", "qz", "qw"]].to_numpy()
    else:
        quat = R.from_rotvec(df[["rx", "ry", "rz"]].to_numpy()).as_quat()
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return t, pos, quat


def gt_relative(cam, marker):
    """marker pose in camera frame, on the camera's timestamps."""
    t = cam.t
    mpos, mquat = mc.resample_to(marker, t)
    Twc = cam.matrices()
    Twm = np.tile(np.eye(4), (len(t), 1, 1))
    Twm[:, :3, :3] = R.from_quat(mquat).as_matrix()
    Twm[:, :3, 3] = mpos
    Tcm = mc.relative_pose(Twc, Twm)
    return t, Tcm[:, :3, 3], R.from_matrix(Tcm[:, :3, :3]).as_quat()


def best_offset(t_gt, d_gt, t_e, d_e, max_off=4.0, step=0.01):
    best = (np.inf, 0.0)
    for o in np.arange(-max_off, max_off, step):
        di = np.interp(t_gt, t_e + o, d_e, left=np.nan, right=np.nan)
        m = np.isfinite(di)
        if m.sum() < 20:
            continue
        e = np.sqrt(np.mean((di[m] - d_gt[m]) ** 2))
        if e < best[0]:
            best = (e, o)
    return best[1]


def evaluate(tag, t_gt, pos_gt, quat_gt, t_e, pos_e, quat_e, frozen=None):
    d_gt = np.linalg.norm(pos_gt, axis=1)
    d_e = np.linalg.norm(pos_e, axis=1)
    off = best_offset(t_gt, d_gt, t_e, d_e)
    t_e2 = t_e + off
    m = (t_e2 >= t_gt[0]) & (t_e2 <= t_gt[-1])
    if frozen:
        m &= mc.valid_time_mask(t_e2, frozen)
    t_e2, pos_e2, d_e2 = t_e2[m], pos_e[m], d_e[m]

    tq = np.clip(t_e2, t_gt[0], t_gt[-1])
    g_pos = np.column_stack([np.interp(tq, t_gt, pos_gt[:, i]) for i in range(3)])
    g_d = np.linalg.norm(g_pos, axis=1)

    corr = np.corrcoef(d_e2, g_d)[0, 1]
    A = np.vstack([g_d, np.ones_like(g_d)]).T
    slope, icpt = np.linalg.lstsq(A, d_e2, rcond=None)[0]

    # best-fit rigid (Umeyama) residual
    mu_g, mu_e = g_pos.mean(0), pos_e2.mean(0)
    U, _, Vt = np.linalg.svd((pos_e2 - mu_e).T @ (g_pos - mu_g))
    dd = np.sign(np.linalg.det(U @ Vt))
    Rb = U @ np.diag([1, 1, dd]) @ Vt
    pos_corr = (Rb @ g_pos.T).T + (mu_e - Rb @ mu_g)
    res = np.linalg.norm(pos_e2 - pos_corr, axis=1)

    return dict(tag=tag, offset=off, corr=corr, slope=slope,
                res_med=np.median(res), res_mean=res.mean(),
                t=t_e2, d_e=d_e2, g_d=g_d,
                est_rng=(d_e2.min(), d_e2.max()), gt_rng=(g_d.min(), g_d.max()))


def main():
    trajs = mc.read_rigid_bodies(BAG)
    t_e, pos_e, quat_e = load_aruco(OUT / "aruco_estimates.csv")
    print(f"loaded {len(t_e)} ArUco rows; estimated |t| range "
          f"[{np.linalg.norm(pos_e,axis=1).min():.2f},"
          f"{np.linalg.norm(pos_e,axis=1).max():.2f}] m")

    results = []
    for cam_id, mk_id in [("2", "1"), ("1", "2")]:
        tg, pg, qg = gt_relative(trajs[cam_id], trajs[mk_id])
        frozen = mc.frozen_intervals(trajs[cam_id])
        r = evaluate(f"cam={cam_id}, marker={mk_id}",
                     tg, pg, qg, t_e, pos_e, quat_e, frozen=frozen)
        results.append(r)

    print("\n  hypothesis           offset   dist-corr  scale   rigid-residual")
    for r in results:
        print(f"  {r['tag']:18s}  {r['offset']:+5.2f}s   {r['corr']:+.3f}    "
              f"{r['slope']:.3f}    med={r['res_med']:.3f} m mean={r['res_mean']:.3f} m")

    best = min(results, key=lambda r: r["res_med"])
    print(f"\n  -> best role assignment: {best['tag']} "
          f"(smallest residual)")
    print(f"     estimated distance range: {best['est_rng'][0]:.2f}..{best['est_rng'][1]:.2f} m")
    print(f"     groundtruth distance range: {best['gt_rng'][0]:.2f}..{best['gt_rng'][1]:.2f} m")
    if abs(best["slope"] - 1) > 0.15:
        print(f"     [!] scale slope {best['slope']:.2f} != 1  -> check MARKER_SIZE / focal length")
    if best["corr"] < 0.8:
        print(f"     [!] low distance correlation -> likely WRONG video FPS (clock-rate mismatch)")

    # figure: distance overlay for the better hypothesis
    fig, ax = plt.subplots(2, 1, figsize=(9, 7))
    ax[0].plot(best["t"], best["g_d"], "k", lw=0.8, label="ground truth")
    ax[0].plot(best["t"], best["d_e"], "tab:red", lw=0.8, alpha=.8, label="ArUco estimate")
    ax[0].set_ylabel("marker distance [m]"); ax[0].legend()
    ax[0].set_title(f"Distance overlay  ({best['tag']}, corr={best['corr']:.3f})")
    ax[0].grid(alpha=.3); ax[0].set_xlabel("time [s]")
    ax[1].scatter(best["g_d"], best["d_e"], s=4, alpha=.4)
    lim = [0, max(best["g_d"].max(), best["d_e"].max()) * 1.05]
    ax[1].plot(lim, lim, "k--", lw=1, label="ideal y=x")
    ax[1].set_xlabel("ground-truth distance [m]"); ax[1].set_ylabel("estimated distance [m]")
    ax[1].set_title(f"scale slope = {best['slope']:.3f}"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_diagnose.png"); plt.close(fig)
    print(f"\n  wrote {OUT/'fig_diagnose.png'}")


if __name__ == "__main__":
    main()
