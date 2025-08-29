# 전체 실행 흐름 제어
# nodes/executor.py
import numpy as np
import time
from pymycobot import MyCobot280
from utils.command_parser import parse_command
from utils.pixel_to_cam_coords import detect_objects
from utils.camera import capture_d455_images, load_intrinsics
from utils.cam2base_eyetohand import transform_xyz_cam2base, transform_rpy_cam2base
from utils.pointcloud_utils import create_roi_pcd_from_depth, calculate_rpy
from utils.robot_action import pick_and_place

PORT = "/dev/ttyACM2"
BAUD = 115200
GRIPPER_LENGTH_MM = 160
model_path = "/home/ros/llm_robot/yolo/runs/pose/yolo11n_640_500ep/weights/best.pt"
cam2base_path = "/home/ros/llm_robot/data/Calibration/Eye-to-Hand11/cam2base_pgo_inliers.json"

def run():
    # 명령 입력 받기
    # user_input = input("명령어 입력: ")
    
    # # 자연어 명령을 JSON으로 변경
    # cmd = parse_command(user_input)
    # if not cmd:
    #     print("⚠️ 명령 분석 실패")
    #     return
    # print("✅ LLM 명령 파싱 결과:", cmd)

    # 객체 탐지를 위한 이미지 캡쳐 및 뎁스 저장
    print("Intel Real Sense 캡쳐 수행")
    # 로봇 연결 
    # print("1. 로봇 연결 시작")
    # mc = MyCobot280(PORT, BAUD)
    # mc.send_angles([0, 0, 0, -75, 0, 0], 50)
    print("...안정화 진행...")
    time.sleep(1)
    print("2. 캡쳐 수행")
    c_path, d_path, intr_path = capture_d455_images()
    print("3. 캡쳐 완료")

    # 카메라 메트릭스 추출
    camera_matrix, dist_coeffs = load_intrinsics(intr_path)

    # RGB에서 YOLO로 카메라상 좌표 얻기
    print("목표 6-DoF 계산 시작")
    target_list = ["cube"]
    yolo_outputs = detect_objects(c_path, d_path,target_list, 
                                  camera_matrix, dist_coeffs, model_path)
    # Base상 6-DoF    
    cam_xyz = yolo_outputs[0]["cam_xyz"]
    
    # PCA로 RPY 계산 
    pcd = create_roi_pcd_from_depth(d_path, yolo_outputs, camera_matrix)
    roll_normal, pitch_normal, yaw_pca = calculate_rpy(pcd)
    print(f"[Cam] roll_normal: {roll_normal}, pitch_normal: {pitch_normal}, yaw_pca: {yaw_pca}")

    # 선택된 좌표를 hand-eye calibration결과를 활용해 변환
    base_xyz = transform_xyz_cam2base(cam2base_path, cam_xyz)
    base_rpy = transform_rpy_cam2base([roll_normal, pitch_normal, yaw_pca], cam2base_path)
    coords = np.round(base_xyz , 1).tolist() + np.round(base_rpy, 1).tolist()
    print(f"[Base] coords: {coords}")
    # 현재 로봇 환경에 맞는 Hard Coding 추가 후 이동
    print("로봇 이동 시작")
    # pick_and_place(mc, coords)
    

if __name__ == "__main__":
    run()
