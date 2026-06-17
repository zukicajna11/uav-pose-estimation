"""
04_make_video.py
----------------
Renders an MP4 animation of the recording:
  left  : 3D world view, camera moving around the static marker, with a small
          axis-triad showing the camera orientation and a fading trail.
  right : ground-truth marker-camera distance with a moving time cursor.

Usage:  python 04_make_video.py [--fps 30] [--decim 5] [--out output/flythrough.mp4]
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.spatial.transform import Rotation as R

import mocap_rv as mc

BAG = "rv_mocap/rosbag2_2026_06_04-12_53_28"
OUT = Path("output"); OUT.mkdir(exist_ok=True)
CAMERA_BODY, MARKER_BODY = "2", "1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--decim", type=int, default=5, help="use every Nth sample")
    ap.add_argument("--out", type=Path, default=OUT / "flythrough.mp4")
    ap.add_argument("--trail", type=int, default=120, help="trail length (frames)")
    args = ap.parse_args()

    trajs = mc.read_rigid_bodies(BAG)
    cam, marker = trajs[CAMERA_BODY], trajs[MARKER_BODY]

    idx = np.arange(0, len(cam), args.decim)
    t = cam.t[idx]
    pos = cam.pos[idx]
    Rcam = cam.rotations()[idx].as_matrix()          # (M,3,3)
    mk = marker.pos.mean(0)
    gt = np.load(OUT / "groundtruth.npz")
    dist = gt["dist"][idx]

    # consistent world-frame limits
    allpts = np.vstack([pos, mk])
    lo, hi = allpts.min(0), allpts.max(0)
    ctr, rng = (lo + hi) / 2, (hi - lo).max() / 2 * 1.15
    lims = np.column_stack([ctr - rng, ctr + rng])

    fig = plt.figure(figsize=(12, 5.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    axd = fig.add_subplot(1, 2, 2)

    # static marker
    ax.scatter(*mk, color="red", s=120, marker="*", label="ArUco marker")
    trail, = ax.plot([], [], [], color="tab:blue", lw=1.0, alpha=0.7)
    dot, = ax.plot([], [], [], "o", color="tab:blue", ms=7, label="camera")
    # orientation triad (3 quiver-like line segments), updated per frame
    triad = [ax.plot([], [], [], color=col, lw=2)[0]
             for col in ("#d62728", "#2ca02c", "#1f77b4")]  # x,y,z
    ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.set_title("Camera flying around the marker"); ax.legend(loc="upper left")

    axd.plot(t, dist, color="k", lw=0.8)
    cursor = axd.axvline(t[0], color="tab:red", lw=1.5)
    pt, = axd.plot([], [], "o", color="tab:red", ms=6)
    axd.set_xlabel("time [s]"); axd.set_ylabel("marker distance [m]")
    axd.set_title("Ground-truth distance"); axd.grid(alpha=.3)

    L = 0.35  # triad arm length [m]

    def update(k):
        s = max(0, k - args.trail)
        trail.set_data(pos[s:k+1, 0], pos[s:k+1, 1])
        trail.set_3d_properties(pos[s:k+1, 2])
        dot.set_data([pos[k, 0]], [pos[k, 1]])
        dot.set_3d_properties([pos[k, 2]])
        for a in range(3):
            end = pos[k] + Rcam[k][:, a] * L
            triad[a].set_data([pos[k, 0], end[0]], [pos[k, 1], end[1]])
            triad[a].set_3d_properties([pos[k, 2], end[2]])
        cursor.set_xdata([t[k], t[k]])
        pt.set_data([t[k]], [dist[k]])
        ax.view_init(elev=22, azim=-60 + 30 * np.sin(k / len(t) * 2 * np.pi))
        return trail, dot, *triad, cursor, pt

    anim = FuncAnimation(fig, update, frames=len(t), interval=1000/args.fps, blit=False)
    args.out.parent.mkdir(exist_ok=True)
    anim.save(args.out, writer=FFMpegWriter(fps=args.fps, bitrate=2400))
    plt.close(fig)
    print(f"wrote {args.out}  ({len(t)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
