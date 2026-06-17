"""
01_parse_and_export.py
----------------------
Reads the mocap bag and writes tidy CSV/NPZ files to ./output/ :

  body_<name>_trajectory.csv   full 6-DOF trajectory of each rigid body
  groundtruth_marker_in_camera.csv   the marker pose expressed in the camera
                                     frame  ->  this is what your ArUco
                                     detector is supposed to reproduce.

Edit CAMERA_BODY / MARKER_BODY below if the auto-detected roles are wrong.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

import mocap_rv as mc

# --------------------------------------------------------------------------- #
_DEFAULT_BAG = "rv_mocap/rosbag2_2026_06_12-13_53_18"
CAMERA_BODY = "1"
MARKER_BODY = "2"
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default=_DEFAULT_BAG, help="path to ROS2 bag folder")
    ap.add_argument("--out", default="output", help="output directory")
    ap.add_argument("--camera-body", default=CAMERA_BODY, help="rigid body ID for camera")
    ap.add_argument("--marker-body", default=MARKER_BODY, help="rigid body ID for marker")
    ap.add_argument("--z-scale", type=float, default=1.0,
                    help="divide all world Z positions by this factor before computing relative pose "
                         "(use to correct Motive Z-axis calibration errors, e.g. --z-scale 2.0)")
    args = ap.parse_args()

    BAG = args.bag
    OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
    cam_body = args.camera_body
    marker_body = args.marker_body
    z_scale = args.z_scale

    trajs = mc.read_rigid_bodies(BAG)
    print("Rigid bodies in bag:", list(trajs.keys()))

    # 1) per-body trajectories -------------------------------------------- #
    for name, tr in trajs.items():
        df = tr.to_frame()
        path = OUT / f"body_{name}_trajectory.csv"
        df.to_csv(path, index=False)
        print(f"  wrote {path}  ({len(df)} rows)")

    cam, marker = trajs[cam_body], trajs[marker_body]

    # 2) ground-truth relative pose: marker expressed in the CAMERA frame -- #
    #    The camera moves, so we evaluate on the camera's timestamps and
    #    interpolate the (nearly static) marker onto them.
    t = cam.t
    m_pos, m_quat = mc.resample_to(marker, t)

    T_world_cam = cam.matrices()                       # (N,4,4)
    T_world_marker = np.tile(np.eye(4), (len(t), 1, 1))
    T_world_marker[:, :3, :3] = R.from_quat(m_quat).as_matrix()
    T_world_marker[:, :3, 3] = m_pos

    if z_scale != 1.0:
        print(f"  [i] applying Z scale correction: dividing world Z by {z_scale}")
        T_world_cam[:, 2, 3] /= z_scale
        T_world_marker[:, 2, 3] /= z_scale

    T_cam_marker = mc.relative_pose(T_world_cam, T_world_marker)   # (N,4,4)

    rel_pos = T_cam_marker[:, :3, 3]
    rel_quat = R.from_matrix(T_cam_marker[:, :3, :3]).as_quat()
    rel_eul = R.from_matrix(T_cam_marker[:, :3, :3]).as_euler("xyz", degrees=True)
    dist = np.linalg.norm(rel_pos, axis=1)

    gt = pd.DataFrame({
        "t": t,
        "t_ns": cam.t_ns,
        "x": rel_pos[:, 0], "y": rel_pos[:, 1], "z": rel_pos[:, 2],
        "qx": rel_quat[:, 0], "qy": rel_quat[:, 1],
        "qz": rel_quat[:, 2], "qw": rel_quat[:, 3],
        "roll_deg": rel_eul[:, 0], "pitch_deg": rel_eul[:, 1], "yaw_deg": rel_eul[:, 2],
        "distance_m": dist,
    })
    gt_path = OUT / "groundtruth_marker_in_camera.csv"
    gt.to_csv(gt_path, index=False)
    print(f"  wrote {gt_path}  ({len(gt)} rows)")
    print(f"  marker-camera distance: mean={dist.mean():.3f} m  "
          f"range=[{dist.min():.3f}, {dist.max():.3f}]")

    # compact binary for fast reload
    frozen = mc.frozen_intervals(cam)
    np.savez(OUT / "groundtruth.npz",
             t=t, t_ns=cam.t_ns, rel_pos=rel_pos, rel_quat=rel_quat, dist=dist,
             frozen=np.array(frozen, dtype=float).reshape(-1, 2))
    print(f"  wrote {OUT/'groundtruth.npz'}")
    if frozen:
        total = sum(b - a for a, b in frozen)
        print(f"  [!] camera frozen/occluded in {len(frozen)} windows "
              f"(~{total:.1f}s total) -> excluded from comparison")


if __name__ == "__main__":
    main()
