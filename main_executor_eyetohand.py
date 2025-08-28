# 전체 실행 흐름 제어
# nodes/executor.py
import cv2
import time
import numpy as np
from pymycobot import MyCobot280
from utils.command_parser import parse_command
from utils.pixel_to_cam_coords import detect_objects
from utils.camera import capture_d455_images, load_intrinsics
from utils.cam2base_eyetohand import cam2base
from utils.grasp_orientation import get_roi_depth_intrinsic, create_pcd_from_depth, segment_plane_from_pointcloud, pcd_to_surface_normal
from utils.robot_action import move_to_object


PORT = "/dev/ttyACM0"
BAUD = 115200
GRIPPER_LENGTH_MM = 160
ee2cam_path = "/home/ros/llm_robot/data/Calibration/ee2cam_opt3.json"

def run():
    # 명령 입력 받기
    user_input = input("명령어 입력: ")
    
    # 자연어 명령을 JSON으로 변경
    cmd = parse_command(user_input)
    if not cmd:
        print("⚠️ 명령 분석 실패")
        return
    print("✅ LLM 명령 파싱 결과:", cmd)

    # 객체 탐지를 위한 이미지 캡쳐 및 뎁스 저장
    print("Intel Real Sense 캡쳐 수행")
    # 로봇 연결 
    print("1. 로봇 연결 시작")
    mc = MyCobot280(PORT, BAUD)
    mc.send_angles([0, 0, 0, -75, 0, 0], 50)
    print("...안정화 진행...")
    time.sleep(1)
    print("2. 캡쳐 수행")
    c_path, d_path, intr_path = capture_d455_images()
    coords = mc.get_coords()
    print("3. 캡쳐 완료")
    print("Saved:", c_path, d_path)

    # 카메라 메트릭스 추출
    camera_matrix, dist_coeffs = load_intrinsics(intr_path)

    # RGB에서 YOLO로 카메라상 좌표 얻기
    best_model = "/home/ros/llm_robot/yolo11_seg_cube_best.pt"
    target_list = [t["name"] for t in cmd["targets"]]
    yolo_outputs = detect_objects(c_path, d_path,target_list, 
                                  camera_matrix, dist_coeffs, best_model)
    
    cam_xyz = yolo_outputs[0]["cam_xyz"]
    
    raw_depth = np.load(d_path)
    roi_depth, roi_intrinsic = get_roi_depth_intrinsic(raw_depth, yolo_outputs, intr_path)
    pcd = create_pcd_from_depth(roi_depth, roi_intrinsic)
    pcd = segment_plane_from_pointcloud(pcd)
    rotation_matrix, yaw_angle = pcd_to_surface_normal(pcd)
    
    # 선택된 좌표를 hand-eye calibration결과를 활용해 변환
    base_coords = cam2base(cam_xyz, coords, rotation_matrix, yaw_angle, 
                        ee2cam_path, ee2cam = False)

    # 현재 로봇 환경에 맞는 Hard Coding 추가 후 이동
    # move_to_object(mc, base_xyz_mm, gripper_length_mm=GRIPPER_LENGTH_MM)
    



if __name__ == "__main__":
    run()
