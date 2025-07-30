"""
robot_actions.py
─────────────────────────────────────────────────────
MyCobot 280 M5 전용 ― MoveIt 없이 파이썬 API만으로
 • Hand-Eye 행렬 로드
 • 카메라 XYZ → 베이스 좌표 변환
 • Top-grasp(위에서 집기) 시퀀스 실행
─────────────────────────────────────────────────────
"""
import json, time
import numpy as np
from scipy.spatial.transform import Rotation as R
from pymycobot import MyCobot280


SERIAL_PORT = "/dev/ttyACM2"
BAUD_RATE = 115200
HAND_EYE_PATH = "/home/ros/llm_robot/data/Calibration/handeye_result.yaml"

GRIPPER_OPEN  = 100        # 상황에 맞게 조정
GRIPPER_CLOSE =   0
MOVE_SPEED         = 50         # 0~100
HOVER_Z = 0.05                  # M
DESCEND_OFFSET = 0.005


mc = MyCobot280(SERIAL_PORT, BAUD_RATE)
with open(HAND_EYE_PATH) as f:
    H = json.load(f)
    
_R = np.array(H["rotation"])
_t = np.array(H["translation"]).reshape(3, 1)
_T_b_c = np.vstack([np.hstack([np.array(H["rotation"]),
                              np.array(H["translation"]).reshape(3, 1)]),
                   [0, 0, 0, 1]])

# --- 내부 유틸 ---
def cam_to_base(x_cam: float, y_cam: float, z_cam: float):
    """카메라 좌표 → 베이스 좌표 (m)."""
    p_base = _T_b_c @ np.array([x_cam, y_cam, z_cam, 1.0])
    return p_base[:3]                      # ndarray(3,)

def coords_to_mm(x:float, y:float, z: float,
                roll:float, pitch: float, yaw:float):
    """Move-coords 배열 형식 [x mm, y mm, z mm, r deg, p deg, y deg]
       Hand-Eye 변환으로 얻은 베이스 좌표는 M단위이고 cobot은 mm라서 변환이 필요 
    """
    return [x * 1000, y * 1000, z *1000,
            np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)]

# --- 공개 API ---
def pick_xyz_cam(xb: float, yb: float, zb: float,
                 hover: float = HOVER_Z):
    """
    카메라 XYZ [m] 를 받아 위에서 내려잡기 실행.
    • hover [m] : pre-grasp 높이 (기본 0.05 m)
    """

    # xb, yb, zb = cam_to_base(xc, yc, zc)

    # TCP를 뒤집어서 그리퍼가 바닥을 향하게 함. 
    roll, pitch, yaw = np.pi, 0.0, 0.0
    
    # 목표 지점보다 hover만큼 위에 위치하여 충돌 방지
    pre  = coords_to_mm(xb, yb, zb + hover, roll, pitch, yaw)
    # 실제 집을 위치인 z를 5mm더 내려 잡음 (물체에 접근하여 살짝 내려 그리퍼를 닫고 위로 복귀하는 과정)
    grab = coords_to_mm(xb, yb, zb + DESCEND_OFFSET, roll, pitch, yaw)

    # (1) 그리퍼 열고 접근
    mc.set_gripper_value(GRIPPER_OPEN, 80)
    time.sleep(0.4)

    mc.sync_send_coords(pre,  MOVE_SPEED, 1)
    mc.sync_send_coords(grab, MOVE_SPEED, 1)

    # (2) 집기
    mc.set_gripper_value(GRIPPER_CLOSE, 80)
    time.sleep(0.6)

    # (3) 위로 복귀
    mc.sync_send_coords(pre, MOVE_SPEED, 1)


def move_to_cam_xyz(xc: float, yc: float, zc: float,
                    roll: float = np.pi, pitch: float = 0.0, yaw: float = 0.0,
                    speed: int = MOVE_SPEED, wait: int = 1):
    """
    Hand-Eye 변환 후 해당 위치로 이동만 하는 헬퍼.
    (집기 없이 자리 맞추기용)
    """
    xb, yb, zb = cam_to_base(xc, yc, zc)
    coords = coords_to_mm(xb, yb, zb, roll, pitch, yaw)
    mc.sync_send_coords(coords, speed, wait)