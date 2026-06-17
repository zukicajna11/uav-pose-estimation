"""
05_analysis.py
--------------
Deep evaluation of the ArUco detector against mocap ground truth, AFTER the
scale/roles are correct. Produces output/fig_analysis.png with four panels:

  (a) translation error vs ground-truth distance   -> how depth hurts accuracy
  (b) rotation error vs viewing obliquity           -> where pose flips happen
  (c) rotation-error histogram                       -> the flip population (~180 deg)
  (d) translation error: raw vs EMA-filtered         -> how much smoothing helps

and prints a summary: noise floor (flip-free median errors), flip rate, and the
error reduction an EMA filter would give. Constant frame offsets are removed
robustly (flips excluded) before scoring so the numbers reflect real accuracy.

Run:  python 05_analysis.py [--alpha 0.3]
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R, Slerp

import mocap_rv as mc

BAG = "rv_mocap/rosbag2_2026_06_12-13_53_18"
OUT = Path("output")
CAMERA_BODY, MARKER_BODY = "2", "1"          # confirmed correct by diagnose.py


def load_aruco(path):
    df = pd.read_csv(path)
    # ako je Excel pretvorio tacke u zareze, kolone dodju kao tekst -> probaj decimal=','
    if df[["x", "y", "z"]].dtypes.eq(object).any():
        df = pd.read_csv(path, decimal=",")
    # prisili numericki tip; sve sto nije broj -> NaN, pa izbaci
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["x", "y", "z"]).reset_index(drop=True)

    t = df["t"].to_numpy(dtype=float) if "t" in df else df["t_ns"].to_numpy(dtype=float) / 1e9
    pos = df[["x", "y", "z"]].to_numpy(dtype=float)
    if {"qx", "qy", "qz", "qw"}.issubset(df.columns):
        quat = df[["qx", "qy", "qz", "qw"]].to_numpy(dtype=float)
    else:
        quat = R.from_rotvec(df[["rx", "ry", "rz"]].to_numpy(dtype=float)).as_quat()
    return t, pos, quat / np.linalg.norm(quat, axis=1, keepdims=True)


def best_offset(t_gt, d_gt, t_e, d_e, max_off=4.0, step=0.01):
    best = (np.inf, 0.0)
    for o in np.arange(-max_off, max_off, step):
        di = np.interp(t_gt, t_e + o, d_e, left=np.nan, right=np.nan)
        m = np.isfinite(di)
        if m.sum() >= 20:
            e = np.sqrt(np.mean((di[m] - d_gt[m]) ** 2))
            if e < best[0]:
                best = (e, o)
    return best[1]


def binned_median(x, y, nb=14):
    edges = np.linspace(x.min(), x.max(), nb + 1)
    idx = np.digitize(x, edges)
    bx, by = [], []
    for k in range(1, nb + 1):
        m = idx == k
        if m.sum() > 4:
            bx.append(x[m].mean()); by.append(np.median(y[m]))
    return np.array(bx), np.array(by)


def ema(arr, alpha):
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.3, help="EMA smoothing factor")
    args = ap.parse_args()

    trajs = mc.read_rigid_bodies(BAG)
    cam, marker = trajs[CAMERA_BODY], trajs[MARKER_BODY]

    # ground-truth marker-in-camera on camera timestamps
    tg = cam.t
    mpos, mquat = mc.resample_to(marker, tg)
    Twc = cam.matrices()
    Twm = np.tile(np.eye(4), (len(tg), 1, 1))
    Twm[:, :3, :3] = R.from_quat(mquat).as_matrix(); Twm[:, :3, 3] = mpos
    Tcm = mc.relative_pose(Twc, Twm)
    pos_gt = Tcm[:, :3, 3]
    quat_gt = R.from_matrix(Tcm[:, :3, :3]).as_quat()

    # load + time-align estimates
    t_e, pos_e, quat_e = load_aruco(OUT / "aruco_estimates.csv")
    off = best_offset(tg, np.linalg.norm(pos_gt, axis=1),
                      t_e, np.linalg.norm(pos_e, axis=1))
    t_e = t_e + off
    m = (t_e >= tg[0]) & (t_e <= tg[-1])
    t_e, pos_e, quat_e = t_e[m], pos_e[m], quat_e[m]

    # drop samples that fall in frozen (occluded) ground-truth windows
    frozen = mc.frozen_intervals(cam)
    keep = mc.valid_time_mask(t_e, frozen)
    dropped = (~keep).sum()
    t_e, pos_e, quat_e = t_e[keep], pos_e[keep], quat_e[keep]
    print(f"  excluded {dropped} samples in {len(frozen)} frozen GT windows "
          f"({dropped/len(keep)*100:.0f}% of data)")

    tq = np.clip(t_e, tg[0], tg[-1])
    g_pos = np.column_stack([np.interp(tq, tg, pos_gt[:, i]) for i in range(3)])
    g_quat = Slerp(tg, R.from_quat(quat_gt))(tq).as_quat()

    # --- remove constant translation offset (Umeyama R,t) --------------- #
    mu_g, mu_e = g_pos.mean(0), pos_e.mean(0)
    U, _, Vt = np.linalg.svd((pos_e - mu_e).T @ (g_pos - mu_g))
    Rb = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    g_pos_a = (Rb @ g_pos.T).T + (mu_e - Rb @ mu_g)
    terr = np.linalg.norm(pos_e - g_pos_a, axis=1)

    # --- robust constant rotation offset (exclude flips) ---------------- #
    Roff = R.from_quat(quat_e) * R.from_quat(g_quat).inv()
    Qb = Roff.mean()
    for _ in range(3):
        rerr_tmp = (R.from_quat(quat_e) * (Qb * R.from_quat(g_quat)).inv()).magnitude()
        inl = np.degrees(rerr_tmp) < 60
        if inl.sum() > 10:
            Qb = Roff[inl].mean()
    rerr = np.degrees((R.from_quat(quat_e) * (Qb * R.from_quat(g_quat)).inv()).magnitude())

    # viewing obliquity from ground truth (0 = face-on, 90 = edge-on)
    normal = R.from_quat(g_quat).apply(np.array([0, 0, 1.0]))
    los = g_pos / np.linalg.norm(g_pos, axis=1, keepdims=True)
    obliq = np.degrees(np.arccos(np.clip(np.abs(np.sum(normal * los, axis=1)), 0, 1)))

    # flips and noise floor
    flip = rerr > 90
    floor_t = np.median(terr[~flip]); floor_r = np.median(rerr[~flip])

    # EMA preview on translation
    order = np.argsort(t_e)
    pe_s = pos_e[order]
    pe_ema = np.column_stack([ema(pe_s[:, i], args.alpha) for i in range(3)])
    terr_ema = np.linalg.norm(pe_ema - g_pos_a[order], axis=1)

    print(f"  time offset: {off:+.2f} s   matched samples: {len(t_e)}")
    print(f"  translation error  : median={np.median(terr):.3f} m  "
          f"mean={terr.mean():.3f} m  p95={np.percentile(terr,95):.3f} m")
    print(f"  rotation error     : median={np.median(rerr):.2f} deg  "
          f"mean={rerr.mean():.2f} deg")
    print(f"  pose flips (>90 deg): {flip.mean()*100:.1f}%  of frames")
    print(f"  NOISE FLOOR (flip-free): trans median={floor_t:.3f} m  "
          f"rot median={floor_r:.2f} deg")
    print(f"  EMA(alpha={args.alpha}) translation: median "
          f"{np.median(terr):.3f} -> {np.median(terr_ema):.3f} m")

    # ---- figure -------------------------------------------------------- #
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    d_gt = np.linalg.norm(g_pos, axis=1)

    ax[0, 0].scatter(d_gt, terr, s=4, alpha=.3, color="tab:red")
    bx, by = binned_median(d_gt, terr); ax[0, 0].plot(bx, by, "k-o", ms=4, label="binned median")
    ax[0, 0].set_xlabel("ground-truth distance [m]"); ax[0, 0].set_ylabel("translation error [m]")
    ax[0, 0].set_title("(a) translation error vs distance"); ax[0, 0].grid(alpha=.3); ax[0, 0].legend()

    ax[0, 1].scatter(obliq, rerr, s=4, alpha=.3, color="tab:purple")
    bx, by = binned_median(obliq, rerr); ax[0, 1].plot(bx, by, "k-o", ms=4, label="binned median")
    ax[0, 1].set_xlabel("viewing obliquity [deg]  (0=face-on)"); ax[0, 1].set_ylabel("rotation error [deg]")
    ax[0, 1].set_title("(b) rotation error vs viewing angle"); ax[0, 1].grid(alpha=.3); ax[0, 1].legend()

    ax[1, 0].hist(rerr, bins=60, color="tab:purple", alpha=.8)
    ax[1, 0].axvline(90, color="k", ls="--", label="flip threshold")
    ax[1, 0].set_xlabel("rotation error [deg]"); ax[1, 0].set_ylabel("count")
    ax[1, 0].set_title(f"(c) rotation-error histogram  (flips: {flip.mean()*100:.0f}%)")
    ax[1, 0].legend()

    ax[1, 1].plot(t_e[order], terr[order], lw=0.6, alpha=.5, color="tab:red", label="raw")
    ax[1, 1].plot(t_e[order], terr_ema, lw=0.9, color="tab:blue", label=f"EMA a={args.alpha}")
    ax[1, 1].set_xlabel("time [s]"); ax[1, 1].set_ylabel("translation error [m]")
    ax[1, 1].set_title("(d) raw vs EMA-filtered translation error"); ax[1, 1].grid(alpha=.3); ax[1, 1].legend()

    fig.tight_layout(); fig.savefig(OUT / "fig_analysis.png"); plt.close(fig)
    print(f"\n  wrote {OUT/'fig_analysis.png'}")


if __name__ == "__main__":
    main()
