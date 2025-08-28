import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import Tuple, List, Optional

# results[0].show()
def detect_objects(c_path, d_path, target_list, 
                   camera_matrix, dist_coeffs,  
                   best_model = "/home/ros/llm_robot/yolo/yolo11x.pt"):
    
    depth_raw  = np.load(d_path)
    depth_scale = 0.001    # mm -> m
    
    # ▶ intrinsics
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    ppx = camera_matrix[0, 2]
    ppy = camera_matrix[1, 2]
    
    model = YOLO(best_model)  
    results = model(c_path)
    
    r = results[0]          # 한 장이니까
    xyxy  = r.boxes.xyxy.cpu().numpy()  # [N,4]
    cls   = r.boxes.cls.cpu().numpy().astype(int)
    conf  = r.boxes.conf.cpu().numpy()
    names = r.names                     # {idx:'label'}

    outputs = []  # 최종 (label, cx, cy, z, x_cam, y_cam, z_cam) 리스트

    for box, c, p in zip(xyxy, cls, conf):
        label = names[c]
        if label not in target_list:        # 🎯 원하는 클래스만 통과
            continue

        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Depth ROI → Z (m 기준)
        roi = depth_raw[y1:y2, x1:x2]
        valid = roi[roi > 0]
        if valid.size == 0:
            print(f"[WARN] {label} depth invalid, skip")
            continue
        
        # 평균에서 최소값(평면)을 뺀 값
        z = np.median(valid) * depth_scale
        
        # 픽셀 좌표 보정
        pixel = np.array([[[cx, cy]]], dtype=np.float32)    # (1, 1, 2)
        undistorted = cv2.undistortPoints(pixel, camera_matrix, dist_coeffs)
        cx_u, cy_u = undistorted[0][0]
        
        # 픽셀 → 카메라 XYZ 
        x_cam = round(cx_u * z, 4)
        y_cam = round(cy_u * z, 4)
        z_cam = round(z, 4)

        outputs.append(
            {"label": label,
            "bbox": [x1, y1, x2, y2],
            "pixel_xy": [cx, cy],
            "depth_m": z,
            "cam_xyz": [x_cam, y_cam, z_cam],
            "undistroted_xyz": [float(cx_u), float(cy_u), float(z_cam)], 
            "conf": float(p)}
        )

    print("\n📦 추출된 객체 정보")
    for obj in outputs:
        print(f"{obj['label']:10s}  z={obj['depth_m']:.3f} m  "
            f"cam=({obj['cam_xyz'][0]:.3f}, {obj['cam_xyz'][1]:.3f}, {obj['cam_xyz'][2]:.3f})")
    
    return outputs
        
        

def get_clicked_point_cam_xyz(c_path: str, d_path: str, camera_matrix: np.ndarray, 
                             dist_coeffs: np.ndarray, depth_scale: float = 0.001) -> Optional[List[float]]:
    """
    이미지에서 클릭한 점의 카메라 좌표계 XYZ를 구하는 함수
    
    Args:
        c_path (str): 컬러 이미지 파일 경로
        d_path (str): 깊이 이미지 파일 경로 (.npy)
        camera_matrix (np.ndarray): 카메라 내부 파라미터 행렬 (3x3)
        dist_coeffs (np.ndarray): 왜곡 계수
        depth_scale (float): 깊이 스케일 (기본값: 0.001, mm -> m 변환)
    
    Returns:
        Optional[List[float]]: 카메라 좌표계 XYZ [x, y, z] (미터 단위), 클릭하지 않으면 None
    """
    
    # 이미지와 깊이 데이터 로드
    color_img = cv2.imread(c_path)
    if color_img is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {c_path}")
    
    depth_raw = np.load(d_path)
    
    # 클릭한 점을 저장할 변수
    clicked_point = None
    
    def mouse_callback(event: int, x: int, y: int, flags: int, param) -> None:
        """마우스 클릭 콜백 함수"""
        nonlocal clicked_point
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point = (x, y)
            print(f"클릭한 점: ({x}, {y})")
    
    # 윈도우 생성 및 마우스 콜백 설정
    window_name = "Click to get camera XYZ"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # 이미지 표시
    cv2.imshow(window_name, color_img)
    print("이미지에서 원하는 점을 클릭하세요. ESC를 누르면 종료됩니다.")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # ESC 키로 종료
        if key == 27:
            break
        
        # 클릭한 점이 있으면 처리
        if clicked_point is not None:
            x, y = clicked_point
            
            # 깊이값 가져오기 (ROI 주변 평균 사용)
            roi_size = 5
            y1, y2 = max(0, y - roi_size), min(depth_raw.shape[0], y + roi_size + 1)
            x1, x2 = max(0, x - roi_size), min(depth_raw.shape[1], x + roi_size + 1)
            
            roi = depth_raw[y1:y2, x1:x2]
            valid_depths = roi[roi > 0]
            
            if valid_depths.size == 0:
                print(f"[경고] 점 ({x}, {y})에서 유효한 깊이값을 찾을 수 없습니다.")
                clicked_point = None
                continue
            
            # 깊이값 계산 (중간값 사용)
            z_mm = np.median(valid_depths)
            z_m = z_mm * depth_scale
            
            # 카메라 내부 파라미터
            # fx = camera_matrix[0, 0]
            # fy = camera_matrix[1, 1]
            # ppx = camera_matrix[0, 2]
            # ppy = camera_matrix[1, 2]
            
            # 왜곡 보정
            pixel = np.array([[[x, y]]], dtype=np.float32)
            undistorted = cv2.undistortPoints(pixel, camera_matrix, dist_coeffs)
            x_u, y_u = undistorted[0][0]
            
            # 픽셀 좌표를 카메라 좌표계로 변환
            x_cam = x_u * z_m 
            y_cam = y_u * z_m
            z_cam = z_m
            
            outputs = []
            outputs.append(
                {"pixel_xy": [x_u, y_u],
                "depth_m": z_m,
                "cam_xyz": np.round([x_cam, y_cam, z_cam], 3).tolist(),
                "undistroted_xyz": np.round([x_u, y_u, z_cam], 3).tolist()}
            )
            # 결과 출력
            print(f"\n📊 클릭한 점의 카메라 좌표:")
            print(f"   픽셀 좌표: ({x}, {y})")
            print(f"   왜곡보정 픽셀: ({x_u:.2f}, {y_u:.2f})")
            print(f"   깊이: {z_m:.3f} m ({z_mm:.1f} mm)")
            print(f"   카메라 XYZ: ({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f}) m")
            
            cv2.destroyAllWindows()
            return outputs
    
    cv2.destroyAllWindows()
    return None