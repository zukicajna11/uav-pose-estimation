"""
evaluate_dlt.py
---------------
Pure DLT-based Pose Estimation for UAV Landing.
Sve se snima u folder 'output/'.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from aruco_logger import ArucoLogger

# --- Load Calibration Data ---
calib = np.load("calib_data.npz")
K     = calib["K"]
dist  = calib["dist"]
K_inv = np.linalg.inv(K)

# --- Constants ---
MARKER_SIZE = 0.17        
VIDEO_ULAZ  = "uav_landing_rgb_20260612_133834.mp4"
TARGET_ID   = None

# --- ArUco Detector Setup ---
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parametri  = cv2.aruco.DetectorParameters()
detektor   = cv2.aruco.ArucoDetector(aruco_dict, parametri)

class PoseKalman:
    def __init__(self, fps, q_pos=1e-4, q_vel=1e-2, r_meas=1e-2):
        dt = 1.0 / fps
        self._kf_t = self._build(dt, q_pos, q_vel, r_meas)
        self._kf_r = self._build(dt, q_pos, q_vel, r_meas)
        self.initialized = False

    @staticmethod
    def _build(dt, q_pos, q_vel, r_meas):
        kf = cv2.KalmanFilter(6, 3, 0, cv2.CV_64F)
        kf.transitionMatrix = np.array([
            [1, 0, 0, dt, 0,  0 ], [0, 1, 0, 0,  dt, 0 ], [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0 ], [0, 0, 0, 0,  1,  0 ], [0, 0, 0, 0,  0,  1 ],
        ], dtype=np.float64)
        kf.measurementMatrix = np.eye(3, 6, dtype=np.float64)
        kf.processNoiseCov = np.diag([q_pos]*3 + [q_vel]*3)
        kf.measurementNoiseCov = np.eye(3, dtype=np.float64) * r_meas
        kf.errorCovPost = np.eye(6, dtype=np.float64)
        return kf

    def step(self, t=None, rvec=None):
        if not self.initialized:
            if t is None: return None, None
            self._kf_t.statePost = np.array([*t, 0, 0, 0], dtype=np.float64).reshape(6, 1)
            self._kf_r.statePost = np.array([*rvec, 0, 0, 0], dtype=np.float64).reshape(6, 1)
            self.initialized = True
            return t.copy(), rvec.copy()
        pt, pr = self._kf_t.predict(), self._kf_r.predict()
        if t is not None:
            out_t = self._kf_t.correct(t.reshape(3, 1))[:3, 0]
            out_r = self._kf_r.correct(rvec.reshape(3, 1))[:3, 0]
        else:
            out_t, out_r = pt[:3, 0], pr[:3, 0]
        return out_t, out_r

def get_marker_points_3d(marker_size):
    h = marker_size / 2
    return np.array([[-h,h,0], [h,h,0], [h,-h,0], [-h,-h,0]], dtype=np.float64)

def hartley_normalization(pts):
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2) / (d + 1e-10)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    pts_n = (T @ np.column_stack([pts, np.ones(len(pts))]).T).T[:, :2]
    return pts_n, T

def dlt_homography(pts_3d_xy, pts_2d):
    pts_3d_n, T1 = hartley_normalization(pts_3d_xy)
    pts_2d_n, T2 = hartley_normalization(pts_2d)
    A = np.zeros((2 * len(pts_3d_n), 9))
    for i in range(len(pts_3d_n)):
        X, Y = pts_3d_n[i]
        u, v = pts_2d_n[i]
        A[2*i]     = [ X,  Y,  1,   0,  0,  0,  -u*X, -u*Y, -u]
        A[2*i + 1] = [ 0,  0,  0,   X,  Y,  1,  -v*X, -v*Y, -v]
    _, _, Vt = np.linalg.svd(A)
    H = np.linalg.inv(T2) @ Vt[-1].reshape(3, 3) @ T1
    return H

def decompose_homography(H, K_inv):
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    lambda_scale = 1.0 / np.linalg.norm(K_inv @ h1)
    if (lambda_scale * (K_inv @ h3)[2]) < 0: lambda_scale = -lambda_scale
    r1, r2 = lambda_scale * (K_inv @ h1), lambda_scale * (K_inv @ h2)
    t = lambda_scale * (K_inv @ h3)
    R_approx = np.column_stack([r1, r2, np.cross(r1, r2)])
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R, t

def calculate_reprojection_error(pts_3d, pts_2d, R, t, K):
    Rt = np.column_stack([R, t])
    errors = []
    for X_3d, u_meas in zip(pts_3d, pts_2d):
        p_cam = Rt @ np.append(X_3d, 1.0)
        if p_cam[2] <= 0: continue
        p_img = K @ p_cam
        errors.append(np.linalg.norm((p_img[:2]/p_img[2]) - u_meas))
    return np.mean(errors) if errors else np.inf

def draw_axes(img, R, t, K, length=0.05):
    def proj(P):
        p = K @ (R @ P + t)
        if p[2] <= 0: return None
        return (int(round(p[0]/p[2])), int(round(p[1]/p[2])))
    origin = proj(np.zeros(3))
    px, py, pz = proj(np.array([length,0,0])), proj(np.array([0,length,0])), proj(np.array([0,0,length]))
    if origin:
        if px: cv2.arrowedLine(img, origin, px, (0, 0, 255), 3)
        if py: cv2.arrowedLine(img, origin, py, (0, 255, 0), 3)
        if pz: cv2.arrowedLine(img, origin, pz, (255, 0, 0), 3)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=VIDEO_ULAZ)
    ap.add_argument("--kalman", action="store_true")
    args = ap.parse_args()

    # --- Directory Setup ---
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    video_izlaz_putanja = str(out_dir / "rezultat.avi") # SAD JE EKSPLICITNO U OUTPUT/

    cap = cv2.VideoCapture(args.video)
    sirina = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    visina = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_izlaz_putanja, fourcc, fps, (sirina, visina))

    log_raw = ArucoLogger(str(out_dir / "aruco_estimates.csv"))
    
    pts_3d = get_marker_points_3d(MARKER_SIZE)
    pts_3d_xy = pts_3d[:, :2]
    frame_idx = 0

    if args.kalman:
        kf = PoseKalman(fps)
        log_kf = ArucoLogger(str(out_dir / "aruco_estimates_kalman.csv"))

    while True:
        ret, frame = cap.read()
        if not ret: break

        timestamp = frame_idx / fps
        frame_idx += 1
        
        frame_undist = cv2.undistort(frame, K, dist)
        corners, ids, _ = detektor.detectMarkers(frame_undist)
        best_pose = None

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame_undist, corners, ids)
            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                pts_2d = corners[i][0].astype(np.float64)

                try:
                    H = dlt_homography(pts_3d_xy, pts_2d)
                    R, t = decompose_homography(H, K_inv)
                    err = calculate_reprojection_error(pts_3d, pts_2d, R, t, K)

                    if err < 25.0:
                        draw_axes(frame_undist, R, t, K, MARKER_SIZE*0.8)
                        cv2.putText(frame_undist, f"ID:{marker_id} err:{err:.1f}px", 
                                    (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                        
                        if TARGET_ID is None or marker_id == TARGET_ID:
                            if best_pose is None or err < best_pose[0]:
                                best_pose = (err, t, R)
                    else:
                        cv2.putText(frame_undist, f"ERR TOO HIGH: {err:.1f}px", 
                                    (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                except: continue

        if best_pose:
            _, t_raw, R_raw = best_pose
            log_raw.add(timestamp, t_raw, cv2.Rodrigues(R_raw)[0])

        if args.kalman:
            t_in = best_pose[1] if best_pose else None
            r_in = cv2.Rodrigues(best_pose[2])[0].flatten() if best_pose else None
            t_kf, r_kf = kf.step(t_in, r_in)
            if t_kf is not None: log_kf.add(timestamp, t_kf, r_kf)

        video_writer.write(frame_undist)
        cv2.imshow("Pure DLT Pose Estimation", frame_undist)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    video_writer.release()
    log_raw.close()
    if args.kalman: log_kf.close()
    cv2.destroyAllWindows()
    print(f"\nSVE GOTOVO!")
    print(f"Video je snimljen u: {video_izlaz_putanja}")
    print(f"Logovi su u: {out_dir}/")

if __name__ == "__main__":
    main()