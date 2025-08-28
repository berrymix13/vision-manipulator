import numpy as np
import json
from scipy.spatial.transform import Rotation

# 오일러각 → 회전행렬
def euler_to_rotation_matrix(rx, ry, rz):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

def cam2base(cam_xyz, coords,
             rotation_matrix, yaw_angle, 
             ee2cam_path, ee2cam = False):
    with open(ee2cam_path, "r") as f:
        data = json.load(f)

    if ee2cam:
        R_ee2cam = np.array(data["R_ee2cam"])
        t_ee2cam = np.array(data["t_ee2cam"]).reshape(3, 1)

        # 변환 행렬 만들기: T_ee2cam (End-Effector → Camera)
        T_ee2cam = np.eye(4)
        T_ee2cam[:3, :3] = R_ee2cam
        T_ee2cam[:3, 3] = t_ee2cam.flatten()
        T_cam2ee = np.linalg.inv(T_ee2cam)
        
    else:
        R_cam2ee = np.array(data["R_cam2ee"])
        R_ee2cam = np.linalg.inv(R_cam2ee)
        t_cam2ee = np.array(data["t_cam2ee"]).reshape(3, 1)

        # 변환 행렬 만들기: T_cam2ee (End-Effector → Camera)
        T_cam2ee = np.eye(4)
        T_cam2ee[:3, :3] = R_cam2ee
        T_cam2ee[:3, 3] = t_cam2ee.flatten()

    # 현재 로봇 포즈 가져오기 (coords 사용)
    position = np.array(coords[0:3]).reshape(3, 1) / 1000.0  # mm → m
    rx, ry, rz = np.radians(coords[3:6])

    R_base2ee = euler_to_rotation_matrix(rx, ry, rz)

    # --- Base XYZ 계산 ---
    # T_base2ee 행렬 구성
    T_base2ee = np.eye(4)
    T_base2ee[:3, :3] = R_base2ee
    T_base2ee[:3, 3] = position.flatten()
    
    cam_xyz = np.array(cam_xyz).reshape(3, 1)
    cam_xyz_h = np.vstack([cam_xyz, [[1]]])  # 4x1 homogeneous vectr

    # base_xyz 계산
    base_xyz_h = T_base2ee @ T_cam2ee @ cam_xyz_h
    base_xyz = base_xyz_h[:3].flatten()*1000  # 최종 좌표 (단위: m)

    # --- Base RPY 계산 ---
    R_base2cam_cur = R_base2ee.dot(R_ee2cam)
    R_base2ee_norm = R_base2cam_cur.dot(rotation_matrix).dot(R_cam2ee)
    
    roll_norm, pitch_norm, _ = Rotation.from_matrix(R_base2ee_norm).as_euler('xyz', degrees=True)

    final_rpy = [roll_norm, pitch_norm, yaw_angle]
    
    # 최종 결과 반환
    combined = np.hstack((base_xyz, final_rpy))
    combined_list = [round(float(x), 1) for x in combined]
    
    return combined_list
