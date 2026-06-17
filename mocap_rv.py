"""
mocap_rv.py
===========
Core utilities for the RV motion-capture / ArUco evaluation project.

Reads a mocap4r2 (OptiTrack) ROS2 bag, extracts the 6-DOF trajectories of the
recorded rigid bodies, and provides pose-algebra helpers used to build the
ground-truth "marker-in-camera" pose that an ArUco detector should reproduce.

No ROS installation is required: the bag is read with the pure-python
`rosbags` library and the custom `mocap4r2_msgs` types are registered inline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from rosbags.typesys.store import Typestore

# --------------------------------------------------------------------------- #
#  mocap4r2_msgs definitions (from the MOCAP4ROS2-Project repository)
# --------------------------------------------------------------------------- #
_MARKER = """
int8 USE_NAME=0
int8 USE_INDEX=1
int8 USE_BOTH=2
int8 id_type
int32 marker_index
string marker_name
geometry_msgs/Point translation
"""
_MARKERS = """
std_msgs/Header header
uint32 frame_number
mocap4r2_msgs/Marker[] markers
"""
_RIGID_BODY = """
string rigid_body_name
mocap4r2_msgs/Marker[] markers
geometry_msgs/Pose pose
"""
_RIGID_BODIES = """
std_msgs/Header header
uint32 frame_number
mocap4r2_msgs/RigidBody[] rigidbodies
"""


def build_typestore() -> Typestore:
    """ROS2 Humble typestore with the mocap4r2 custom messages registered."""
    ts = get_typestore(Stores.ROS2_HUMBLE)
    types = {}
    types.update(get_types_from_msg(_MARKER, "mocap4r2_msgs/msg/Marker"))
    types.update(get_types_from_msg(_MARKERS, "mocap4r2_msgs/msg/Markers"))
    types.update(get_types_from_msg(_RIGID_BODY, "mocap4r2_msgs/msg/RigidBody"))
    types.update(get_types_from_msg(_RIGID_BODIES, "mocap4r2_msgs/msg/RigidBodies"))
    ts.register(types)
    return ts


# --------------------------------------------------------------------------- #
#  Data container
# --------------------------------------------------------------------------- #
@dataclass
class Trajectory:
    """Time-stamped 6-DOF trajectory of a single rigid body.

    Attributes
    ----------
    name : rigid-body id as stored by the mocap system
    t    : (N,)   time in seconds, relative to the start of the recording
    t_ns : (N,)   absolute header timestamp in nanoseconds
    pos  : (N,3)  x, y, z position [m] in the mocap world frame
    quat : (N,4)  orientation as (x, y, z, w) quaternion (scipy convention)
    """
    name: str
    t: np.ndarray
    t_ns: np.ndarray
    pos: np.ndarray
    quat: np.ndarray
    frame_number: np.ndarray = field(default_factory=lambda: np.array([]))

    def __len__(self) -> int:
        return len(self.t)

    # -- convenience ------------------------------------------------------- #
    def rotations(self) -> R:
        return R.from_quat(self.quat)

    def euler_deg(self, seq: str = "xyz") -> np.ndarray:
        """Euler angles in degrees, sequence `seq` (default intrinsic xyz)."""
        return self.rotations().as_euler(seq, degrees=True)

    def matrices(self) -> np.ndarray:
        """(N,4,4) homogeneous transforms world<-body."""
        M = np.tile(np.eye(4), (len(self), 1, 1))
        M[:, :3, :3] = self.rotations().as_matrix()
        M[:, :3, 3] = self.pos
        return M

    def to_frame(self) -> pd.DataFrame:
        e = self.euler_deg()
        return pd.DataFrame({
            "t": self.t,
            "t_ns": self.t_ns,
            "frame_number": self.frame_number,
            "x": self.pos[:, 0], "y": self.pos[:, 1], "z": self.pos[:, 2],
            "qx": self.quat[:, 0], "qy": self.quat[:, 1],
            "qz": self.quat[:, 2], "qw": self.quat[:, 3],
            "roll_deg": e[:, 0], "pitch_deg": e[:, 1], "yaw_deg": e[:, 2],
        })


# --------------------------------------------------------------------------- #
#  Bag reading
# --------------------------------------------------------------------------- #
def read_rigid_bodies(bag_path: str | Path,
                      topic: str = "/rigid_bodies") -> dict[str, Trajectory]:
    """Extract every rigid-body trajectory from the bag.

    Returns a dict {body_name: Trajectory}. Bodies that are momentarily absent
    from a frame simply have no sample for that frame (the time vectors of two
    bodies are therefore not guaranteed to be identical -> use the interpolation
    helpers when comparing them sample by sample).
    """
    bag_path = Path(bag_path)
    ts = build_typestore()

    acc: dict[str, dict[str, list]] = {}
    with Reader(bag_path) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise RuntimeError(f"topic {topic} not found in {bag_path}")
        for conn, _t_ns, raw in reader.messages(connections=conns):
            msg = ts.deserialize_cdr(raw, conn.msgtype)
            stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            for rb in msg.rigidbodies:
                p, o = rb.pose.position, rb.pose.orientation
                # skip invalid / dropped poses (occlusion -> all-zero quaternion)
                qn = o.x * o.x + o.y * o.y + o.z * o.z + o.w * o.w
                if qn < 1e-6:
                    continue
                d = acc.setdefault(rb.rigid_body_name,
                                   {"t_ns": [], "pos": [], "quat": [], "fn": []})
                d["t_ns"].append(stamp_ns)
                d["pos"].append((p.x, p.y, p.z))
                d["quat"].append((o.x, o.y, o.z, o.w))
                d["fn"].append(msg.frame_number)

    if not acc:
        raise RuntimeError("no valid rigid-body samples found")

    t0 = min(min(d["t_ns"]) for d in acc.values())
    out: dict[str, Trajectory] = {}
    for name, d in acc.items():
        t_ns = np.asarray(d["t_ns"], dtype=np.int64)
        order = np.argsort(t_ns)
        t_ns = t_ns[order]
        quat = np.asarray(d["quat"])[order]
        # normalise quaternions defensively
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        out[name] = Trajectory(
            name=name,
            t=(t_ns - t0) / 1e9,
            t_ns=t_ns,
            pos=np.asarray(d["pos"])[order],
            quat=quat,
            frame_number=np.asarray(d["fn"])[order],
        )
    return out


# --------------------------------------------------------------------------- #
#  Pose algebra
# --------------------------------------------------------------------------- #
def pose_to_matrix(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Single homogeneous 4x4 transform from position + (x,y,z,w) quaternion."""
    M = np.eye(4)
    M[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    M[:3, 3] = pos
    return M


def invert(T: np.ndarray) -> np.ndarray:
    """Inverse of a (4,4) or (N,4,4) homogeneous transform."""
    T = np.asarray(T)
    Rm = T[..., :3, :3]
    t = T[..., :3, 3]
    Ti = np.zeros_like(T)
    Ti[..., :3, :3] = np.swapaxes(Rm, -1, -2)
    Ti[..., :3, 3] = -np.einsum("...ij,...j->...i", np.swapaxes(Rm, -1, -2), t)
    Ti[..., 3, 3] = 1.0
    return Ti


def relative_pose(T_world_a: np.ndarray, T_world_b: np.ndarray) -> np.ndarray:
    """Pose of b expressed in frame a:  T_a_b = inv(T_world_a) @ T_world_b."""
    return invert(T_world_a) @ T_world_b


def resample_to(src: Trajectory, t_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a trajectory onto arbitrary query times (seconds).

    Position is linearly interpolated; orientation uses SLERP. Query times
    outside the source span are clamped to the nearest endpoint.
    Returns (pos_interp (M,3), quat_interp (M,4) xyzw).
    """
    from scipy.spatial.transform import Slerp
    t_query = np.clip(t_query, src.t[0], src.t[-1])
    pos = np.column_stack([
        np.interp(t_query, src.t, src.pos[:, i]) for i in range(3)
    ])
    slerp = Slerp(src.t, src.rotations())
    quat = slerp(t_query).as_quat()
    return pos, quat


def angular_distance_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Geodesic angle [deg] between two sets of (N,4) xyzw quaternions."""
    r = R.from_quat(q1) * R.from_quat(q2).inv()
    return np.degrees(r.magnitude())


# --------------------------------------------------------------------------- #
#  Data quality: frozen / stale (occluded) segments
# --------------------------------------------------------------------------- #
def frozen_intervals(traj: "Trajectory", eps: float = 1e-6,
                     pad_s: float = 0.05) -> list[tuple[float, float]]:
    """Time windows where the body is FROZEN (mocap lost it and republished
    the last pose). Detected as runs of consecutive identical positions.

    Returns a list of (t_start, t_end) seconds, padded by `pad_s` on each side.
    During these windows the ground truth is unreliable and should be excluded
    from any comparison.
    """
    if len(traj) < 2:
        return []
    dp = np.linalg.norm(np.diff(traj.pos, axis=0), axis=1)
    stale = dp < eps                      # stale[i] -> sample i+1 repeats sample i
    intervals, i, n = [], 0, len(stale)
    while i < n:
        if stale[i]:
            j = i
            while j < n and stale[j]:
                j += 1
            t0 = traj.t[i] - pad_s
            t1 = traj.t[min(j, len(traj.t) - 1)] + pad_s
            intervals.append((float(t0), float(t1)))
            i = j + 1
        else:
            i += 1
    return intervals


def valid_time_mask(query_t: np.ndarray,
                    intervals: list[tuple[float, float]]) -> np.ndarray:
    """Boolean mask: False where query_t falls inside any frozen interval."""
    query_t = np.asarray(query_t)
    mask = np.ones(len(query_t), dtype=bool)
    for t0, t1 in intervals:
        mask &= ~((query_t >= t0) & (query_t <= t1))
    return mask


# --------------------------------------------------------------------------- #
#  Quick stats
# --------------------------------------------------------------------------- #
def summarize(traj: Trajectory) -> dict:
    dt = np.diff(traj.t)
    span = traj.pos.max(axis=0) - traj.pos.min(axis=0)
    path_len = float(np.linalg.norm(np.diff(traj.pos, axis=0), axis=1).sum())
    return {
        "name": traj.name,
        "n_samples": len(traj),
        "duration_s": float(traj.t[-1] - traj.t[0]),
        "rate_hz": float(1.0 / np.median(dt)) if len(dt) else float("nan"),
        "max_gap_s": float(dt.max()) if len(dt) else float("nan"),
        "pos_mean": traj.pos.mean(axis=0),
        "pos_span_xyz": span,
        "path_length_m": path_len,
    }
