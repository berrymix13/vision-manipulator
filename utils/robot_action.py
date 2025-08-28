"""
robot_actions.py
─────────────────────────────────────────────────────
MyCobot 280 M5 전용 ― MoveIt 없이 파이썬 API만으로
 • Hand-Eye 행렬 로드
 • 카메라 XYZ → 베이스 좌표 변환
 • Top-grasp(위에서 집기) 시퀀스 실행
─────────────────────────────────────────────────────
"""
import time


def pick_and_place(mc, coords, gripper_length_mm=150):
    """
    카메라 XYZ [m] 를 받아 위에서 내려잡기 실행.
    • hover [m] : pre-grasp 높이 (기본 0.05 m)
    """
    base_x, base_y, base_z, base_roll, base_pitch, base_yaw_corr = coords
    base_z = base_z + gripper_length_mm
    
    mc.set_gripper_value(70,50)
    time.sleep(0.5)
    mc.send_coords([base_x, base_y, base_z+80, -base_roll, base_pitch, base_yaw_corr],40,1)
    time.sleep(1.5)
    mc.send_coords([base_x, base_y, base_z, base_roll, base_pitch, base_yaw_corr],50,1)
    time.sleep(1.5)

    # 그리퍼 닫기
    mc.set_gripper_state(1,50)
    time.sleep(1)

    # # 위로 이동
    mc.send_coords([base_x, base_y, base_z+80, base_roll, base_pitch, base_yaw_corr],60,1)
    time.sleep(0.5)
    # 내려놓을 위치로 이동
    mc.send_coords([121.9, -126.0, 281.8, -179.12, -0.18, -114.26],60,1)
    time.sleep(2)
    mc.set_gripper_value(60,50)
    time.sleep(1)

    # 제자리 복귀
    mc.set_gripper_state(1,50)
    mc.send_coords([46.5, -63.7, 421.1, -89.64, 0.26, -90.43],50,1)