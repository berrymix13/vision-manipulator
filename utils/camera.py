import os
import cv2
import numpy as np
import pyrealsense2 as rs
from datetime import datetime
from pathlib import Path
import json
from typing import Tuple, Dict, Any
    
def capture_d455_images(save_dir: str = "/home/ros/llm_robot/data/captures",
                        rgb_size: Tuple[int, int] = (848, 480), 
                        depth_size: Tuple[int, int] = (848, 480)) -> Tuple[str, str, str]:
    """
    RealSense D455에서 RGB + Depth를 동시에 캡처하고 카메라 내부 파라미터를 저장
    
    Args:
        save_dir: 저장 폴더 경로 (없으면 자동 생성)
        rgb_size: RGB 이미지 크기 (width, height)
        depth_size: Depth 이미지 크기 (width, height)
        
    Returns:
        Tuple[str, str, str]: (color_path, depth_path, intrinsics_path)
        
    Raises:
        RuntimeError: 프레임 캡처 실패 시
    """
        
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 서브디렉토리 생성
    (save_path / "color").mkdir(exist_ok=True)
    (save_path / "depth").mkdir(exist_ok=True)
    (save_path / "intrinsics").mkdir(exist_ok=True)

    # 날짜 기반 파일명 생성
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    color_path = save_path / f"color/{stamp}.jpg"
    depth_path = save_path / f"depth/{stamp}.npy"
    intrinsics_path = save_path / f"intrinsics/{stamp}.json"

    # RealSense 파이프라인 설정
    pipe, cfg = rs.pipeline(), rs.config()

    cfg.enable_stream(rs.stream.color,  *rgb_size,   rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth,  *depth_size, rs.format.z16,  30)
    profile = pipe.start(cfg)
    
    # Depthmap 정렬
    align = rs.align(rs.stream.color)
    frames = align.process(pipe.wait_for_frames())
    depth, color = frames.get_depth_frame(), frames.get_color_frame()
    if not depth or not color:
        pipe.stop()
        raise RuntimeError("Failed to capture frames")    
    
    # 카메라 내부 파라미터 추출
    color_intrinsics = color.profile.as_video_stream_profile().intrinsics
    depth_intrinsics = depth.profile.as_video_stream_profile().intrinsics
    
    # 내부 파라미터를 딕셔너리로 구성
    intrinsics_data: Dict[str, Any] = {
        "color_intrinsics": {
            "width": color_intrinsics.width,
            "height": color_intrinsics.height,
            "fx": float(color_intrinsics.fx),  # 초점 거리 x
            "fy": float(color_intrinsics.fy),  # 초점 거리 y
            "ppx": float(color_intrinsics.ppx),  # 주점 x 좌표
            "ppy": float(color_intrinsics.ppy),  # 주점 y 좌표
            "distortion_model": str(color_intrinsics.model),  # enum을 문자열로 변환
            "distortion_coeffs": [float(coeff) for coeff in color_intrinsics.coeffs]  # 렌즈 왜곡 계수
        },
        "depth_intrinsics": {
            "width": depth_intrinsics.width,
            "height": depth_intrinsics.height,
            "fx": float(depth_intrinsics.fx),
            "fy": float(depth_intrinsics.fy),
            "ppx": float(depth_intrinsics.ppx),
            "ppy": float(depth_intrinsics.ppy),
            "distortion_model": str(depth_intrinsics.model),  # enum을 문자열로 변환
            "distortion_coeffs": [float(coeff) for coeff in depth_intrinsics.coeffs]
        }
    }
    
    # Extrinsics (rotation matrix, translation vector) 추출
    try:
        depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        extrinsics = depth_stream.get_extrinsics_to(color_stream)
        intrinsics_data["extrinsics"] = {
            "rotation_matrix": [float(val) for val in extrinsics.rotation],
            "translation_vector": [float(val) for val in extrinsics.translation]
        }
    except Exception as e:
        print(f"Warning: Could not extract extrinsics: {e}")
        intrinsics_data["extrinsics"] = None
    
    # 저장 (RGB = JPEG, Depth = NPY, Intrinsics = JSON)
    color_img = np.asanyarray(color.get_data())            # uint8 (BGR)
    depth_raw = np.asanyarray(depth.get_data())            # uint16 (mm / depth_scale)

    cv2.imwrite(str(color_path), color_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    np.save(str(depth_path), depth_raw)
    
    # JSON으로 내부 파라미터 저장
    with open(intrinsics_path, 'w', encoding='utf-8') as f:
        json.dump(intrinsics_data, f, indent=2, ensure_ascii=False)

    pipe.stop()
    
    return str(color_path), str(depth_path), str(intrinsics_path)


def load_intrinsics(json_path):

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    intr = data["color_intrinsics"]
    fx, fy = intr["fx"], intr["fy"]
    cx, cy = intr["ppx"], intr["ppy"]

    camera_matrix = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype=np.float32)

    dist_coeffs = np.array(intr["distortion_coeffs"], dtype=np.float32)

    return camera_matrix, dist_coeffs