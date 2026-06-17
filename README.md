# Landing Pad Detection and Pose Estimation for UAV Landing

This repository contains the implementation of a vision-based 6D pose estimation system for autonomous UAV landing using ArUco markers and a custom DLT-based homography decomposition algorithm.

## Overview
The goal of this project is to provide a lightweight, infrastructure-independent solution for precision landing in GPS-denied environments. Unlike standard library-based solvers, this implementation recovers the pose from first principles.

## Key Features
- **Marker Detection:** Real-time ArUco marker identification and corner extraction.
- **Pure DLT Implementation:** Manual homography estimation with Hartley normalization for numerical stability.
- **Metric Pose Recovery:** Explicit decomposition of the homography matrix into rotation and translation.
- **Temporal Filtering:** Exponential moving-average (Kalman) filter for jitter removal.
- **Validation:** Performance evaluated against high-precision OptiTrack motion capture ground truth.

## Repository Structure
- `code/`: Python implementation of the DLT estimator and evaluation scripts.
- `calibration/`: Camera intrinsic and distortion parameters.
- `paper/`: IEEE format technical paper and documentation.

## Requirements
- Python 3.x
- OpenCV (cv2)
- NumPy

## Usage
To run the pose estimator on a video stream:
```bash
python evaluate_dlt.py --video your_video.avi --kalman
```
## Authors
- **Adnan Hajrić**
- **Ajna Zukić**
