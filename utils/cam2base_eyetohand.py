import numpy as np
import json
from scipy.spatial.transform import Rotation

def wrap_deg(a):
    """[-180, 180) 범위로 정규화"""
    return (a + 180.0) % 360.0 - 180.0

def transform_xyz_cam2base(cam2base_path, cam_xyz):
    with open(cam2base_path, "r") as f:
        data = json.load(f)

    R_cam2base = np.array(data["R_cam2base"], dtype=float).reshape(3,3)
    t_cam2base = np.array(data["t_cam2base"], dtype=float).reshape(3)

    T_cam2base = np.eye(4)
    T_cam2base[:3,:3] = R_cam2base
    T_cam2base[:3, 3] = t_cam2base

    cam_h = np.r_[np.asarray(cam_xyz, float).reshape(3), 1.0]
    base_h = T_cam2base @ cam_h
    base_xyz = base_h[:3]*1000            # [mm]
    return base_xyz

def transform_rpy_cam2base(rpy_cam_deg, cam2base_json):
    """
    Transforms RPY angles from camera frame to robot base frame.

    Args:
        rpy_cam (list or np.array): RPY angles in camera frame [roll, pitch, yaw] (in radians).
        cam2base_json (str): Path to the cam2base.json file.

    Returns:
        np.array: RPY angles in base frame [roll, pitch, yaw] (in radians).
    """
    # 1. Load the cam2base extrinsic parameters
    with open(cam2base_json, "r") as f:
        data = json.load(f)
    R_cam2base = np.array(data["R_cam2base"], dtype=np.float32).reshape(3,3)

    rpy_cam_rad = np.deg2rad(rpy_cam_deg)
    # 2. Convert camera RPY to a rotation matrix
    # Note: Use 'xyz' for the Roll, Pitch, Yaw order
    rot_cam = Rotation.from_euler('xyz', rpy_cam_rad)
    R_cam = rot_cam.as_matrix()

    # 3. Combine rotations: R_base = R_cam2base * R_cam
    R_base = np.dot(R_cam2base, R_cam)

    # 4. Convert base rotation matrix back to RPY
    rot_base = Rotation.from_matrix(R_base)
    rpy_base = rot_base.as_euler('xyz')
    rpy_base_deg = np.rad2deg(rpy_base)
    rpy_base_deg[2] = wrap_deg(rpy_base_deg[2])
    
    return rpy_base_deg
