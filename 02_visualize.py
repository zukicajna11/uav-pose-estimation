"""
02_visualize.py
---------------
Generates diagnostic figures into ./output/ :

  fig_world_3d.png        both rigid bodies in the mocap world frame
  fig_camera_timeseries.png  camera position & orientation over time
  fig_relative.png        ground-truth marker-in-camera distance + components
  fig_relative_3d.png     marker path as seen from the camera frame
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pandas as pd

import mocap_rv as mc

BAG = "rv_mocap/rosbag2_2026_06_12-13_53_18"
OUT = Path("output"); OUT.mkdir(exist_ok=True)
CAMERA_BODY, MARKER_BODY = "1", "2"

plt.rcParams.update({"figure.dpi": 120, "font.size": 9})


def main():
    trajs = mc.read_rigid_bodies(BAG)
    cam, marker = trajs[CAMERA_BODY], trajs[MARKER_BODY]

    # ---- 1. World-frame 3D trajectories -------------------------------- #
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*cam.pos.T, lw=0.8, color="tab:blue",
            label=f"camera (body {CAMERA_BODY}), moving")
    ax.scatter(*cam.pos[0], color="tab:blue", s=40, marker="o")
    ax.scatter(*marker.pos.mean(0), color="tab:red", s=80, marker="*",
               label=f"marker (body {MARKER_BODY}), static")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.set_title("Mocap world frame: camera path around the static marker")
    ax.legend(loc="upper left")
    try:  # equal aspect (mpl >= 3.6)
        ax.set_box_aspect(np.ptp(np.vstack([cam.pos, marker.pos.mean(0)]), axis=0))
    except Exception:
        pass
    fig.tight_layout(); fig.savefig(OUT / "fig_world_3d.png"); plt.close(fig)

    # ---- 2. Camera position & orientation time series ------------------ #
    eul = cam.euler_deg()
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for i, lab in enumerate("XYZ"):
        axs[0].plot(cam.t, cam.pos[:, i], lw=0.7, label=lab)
    axs[0].set_ylabel("position [m]"); axs[0].legend(ncol=3); axs[0].grid(alpha=.3)
    axs[0].set_title(f"Camera (body {CAMERA_BODY}) pose in world frame")
    for i, lab in enumerate(["roll", "pitch", "yaw"]):
        axs[1].plot(cam.t, eul[:, i], lw=0.7, label=lab)
    axs[1].set_ylabel("euler [deg]"); axs[1].set_xlabel("time [s]")
    axs[1].legend(ncol=3); axs[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_camera_timeseries.png"); plt.close(fig)

    # ---- 3. Ground-truth relative pose (marker in camera) -------------- #
    gt = pd.read_csv(OUT / "groundtruth_marker_in_camera.csv")
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axs[0].plot(gt.t, gt.distance_m, color="k", lw=0.8)
    axs[0].set_ylabel("distance [m]"); axs[0].grid(alpha=.3)
    axs[0].set_title("Ground truth: marker pose expressed in the camera frame")
    for c in "xyz":
        axs[1].plot(gt.t, gt[c], lw=0.7, label=c)
    axs[1].set_ylabel("position [m]"); axs[1].legend(ncol=3); axs[1].grid(alpha=.3)
    for c in ["roll_deg", "pitch_deg", "yaw_deg"]:
        axs[2].plot(gt.t, gt[c], lw=0.7, label=c.split("_")[0])
    axs[2].set_ylabel("orientation [deg]"); axs[2].set_xlabel("time [s]")
    axs[2].legend(ncol=3); axs[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_relative.png"); plt.close(fig)

    # ---- 4. Marker path in the camera frame (3D) ----------------------- #
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(gt.x, gt.y, gt.z, c=gt.t, cmap="viridis", s=3)
    ax.scatter(0, 0, 0, color="red", s=60, marker="^", label="camera origin")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.set_title("Marker position as seen from the camera frame")
    fig.colorbar(sc, ax=ax, shrink=.6, label="time [s]"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig_relative_3d.png"); plt.close(fig)

    print("wrote 4 figures to", OUT)


if __name__ == "__main__":
    main()
