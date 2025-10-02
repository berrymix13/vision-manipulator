#!/usr/bin/env python3
from turtle import done
import cv2
import json
import time
import math
import numpy as np
import pyrealsense2 as rs
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from pymycobot import MyCobot280
from camera import capture_d455_images, load_intrinsics


class EyeToHandCalibrator:
    """
    Eye-to-Hand calibration을 수행하는 클래스
    
    ChArUco 보드를 End-Effector에 고정하고 다양한 포즈에서 이미지를 캡처하여
    외부 고정 카메라와 로봇 End-Effector 간의 변환 행렬을 계산합니다.
    """
    
    def __init__(self, 
                 robot_port: str = "/dev/ttyACM0",
                 robot_baud: int = 115200,
                 save_dir: str = "/home/ros/llm_robot/data/Calibration/Eye-to-Hand2",
                 charuco_squares_x: int = 7,   # 가로 사각형 개수 (columns)
                 charuco_squares_y: int = 5,   # 세로 사각형 개수 (rows)
                 charuco_square_length: float = 28.0,  # mm
                 charuco_marker_length: float = 14.0,  # mm
                 min_charuco_corners: int = 25,  # 최소 ChArUco 코너 개수
                 min_detection_confidence: float = 0.5,  # 최소 검출 신뢰도
                 max_distance_threshold: float = 0.7):  # 최대 거리 임계값 (미터)
        """
        Eye-to-Hand calibration 초기화
        
        Args:
            robot_port: 로봇 시리얼 포트
            robot_baud: 로봇 통신 보드레이트
            save_dir: 캡처된 이미지와 포즈 데이터 저장 디렉토리
            charuco_squares_x: ChArUco 보드 가로 사각형 개수
            charuco_squares_y: ChArUco 보드 세로 사각형 개수
            charuco_square_length: ChArUco 사각형 한 변의 길이 (mm)
            charuco_marker_length: ChArUco 마커 한 변의 길이 (mm)
        """
        self.robot_port = robot_port
        self.robot_baud = robot_baud
        self.save_dir = Path(save_dir)
        self.poses_dir = self.save_dir / "poses"
        self.angles_dir = self.save_dir / "angles"
        
        # ChArUco 보드 파라미터
        self.charuco_squares_x = charuco_squares_x
        self.charuco_squares_y = charuco_squares_y
        self.charuco_square_length = charuco_square_length  # mm 단위 유지
        self.charuco_marker_length = charuco_marker_length  # mm 단위 유지
        
        # 이상치 처리 파라미터
        self.min_charuco_corners = min_charuco_corners
        self.min_detection_confidence = min_detection_confidence
        self.max_distance_threshold = max_distance_threshold
        
        # ChArUco 딕셔너리 및 보드 생성
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.charuco_board = cv2.aruco.CharucoBoard(
            size=(charuco_squares_x, charuco_squares_y),  # (가로, 세로) 순서로 수정
            squareLength=self.charuco_square_length / 1000.0,  # mm를 m로 변환
            markerLength=self.charuco_marker_length / 1000.0,  # mm를 m로 변환
            dictionary=self.aruco_dict
        )
        
        # 로봇 연결
        self.robot = MyCobot280(robot_port, robot_baud)
        
        # 저장 디렉토리 생성
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.poses_dir.mkdir(parents=True, exist_ok=True)
        self.angles_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Eye-to-Hand Calibrator 초기화 완료")
        print(f"ChArUco 크기: {charuco_squares_x}x{charuco_squares_y}")
        print(f"사각형 길이: {charuco_square_length:.1f}mm")
        print(f"마커 길이: {charuco_marker_length:.1f}mm")
        print(f"Eye-to-Hand calibration 모드")
        print(f"이상치 처리 파라미터:")
        print(f"  - 최소 ChArUco 코너: {min_charuco_corners}개")
        print(f"  - 최소 검출 신뢰도: {min_detection_confidence}")
        print(f"  - 최대 거리 임계값: {max_distance_threshold}m")
        print(f"좌표축 변환:")
        print(f"  - 카메라 Z축 → 로봇 X축")
        print(f"  - 카메라 -X축 → 로봇 Y축")
        print(f"  - 카메라 -Y축 → 로봇 Z축")
    

    def detect_charuco_pose(self, image: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        이미지에서 ChArUco 보드의 포즈를 검출합니다.
        cv_test.ipynb의 코드를 참고하여 OpenCV 4.11.0에 맞게 개선된 버전
        """
        # 1. 이미지 전처리 (CLAHE로 대비 향상)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_enhanced = clahe.apply(gray)
        
        # 보드 설정 정보 출력
        # print(f"\rDEBUG: Board config - size: ({self.charuco_squares_x}, {self.charuco_squares_y}), square_length: {self.charuco_square_length:.1f}mm, marker_length: {self.charuco_marker_length:.1f}mm", end="")
        
        # 2. ChArUco 디텍터 생성 (OpenCV 4.11.0 방식)
        charuco_detector = cv2.aruco.CharucoDetector(self.charuco_board)
        
        # 3. ChArUco 코너 및 마커 검출 (전처리된 이미지 사용)
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray_enhanced)
        
        print(f"\rDEBUG: ChArUco corners: {len(charuco_corners) if charuco_corners is not None else 0}, IDs: {len(charuco_ids) if charuco_ids is not None else 0}, markers: {len(marker_corners) if marker_corners is not None else 0}", end="")
        
        # 4. ChArUco 코너가 충분히 검출된 경우 pose 추정 (최소 6개 필요)
        if (charuco_corners is not None and charuco_ids is not None and 
            len(charuco_corners) >= 6 and len(charuco_corners) == len(charuco_ids)):
            
            print(f"\rDEBUG: ChArUco corners detected: {len(charuco_corners)}, proceeding with pose estimation", end="")
            
            # 5. 2D-3D 대응점 추출 (cv_test.ipynb 방식)
            objPoints, imgPoints = self.charuco_board.matchImagePoints(charuco_corners, charuco_ids)
            
            if objPoints is not None and imgPoints is not None and len(objPoints) >= 6:
                # 6. solvePnP를 통해 보드의 3D 포즈 계산
                success, rvec, tvec = cv2.solvePnP(
                    objPoints,
                    imgPoints,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    print(f"\rDEBUG: ChArUco pose estimation successful with {len(objPoints)} points", end="")
                    return rvec, tvec, charuco_corners, charuco_ids
                else:
                    print(f"\rDEBUG: solvePnP failed for ChArUco corners", end="")
            else:
                print(f"\rDEBUG: Not enough 2D-3D correspondences ({len(objPoints) if objPoints is not None else 0} < 6)", end="")
        else:
            corner_count = len(charuco_corners) if charuco_corners is not None else 0
            if corner_count < 6:
                print(f"\rDEBUG: Skipping pose estimation, only {corner_count} corners detected (need >= 6)", end="")
            else:
                print(f"\rDEBUG: ChArUco detection failed, trying ArUco marker-based fallback", end="")
        
        # 7. ChArUco 검출 실패 시 ArUco 마커 기반 fallback
        print(f"\rDEBUG: ChArUco detection failed, trying ArUco marker-based fallback", end="")
        
        # ArUco 마커 검출
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_params.adaptiveThreshWinSizeMin = 3
        aruco_params.adaptiveThreshWinSizeMax = 23
        aruco_params.adaptiveThreshWinSizeStep = 10
        aruco_params.adaptiveThreshConstant = 7
        aruco_params.minMarkerPerimeterRate = 0.03
        aruco_params.maxMarkerPerimeterRate = 4.0
        aruco_params.polygonalApproxAccuracyRate = 0.03
        aruco_params.minCornerDistanceRate = 0.05
        aruco_params.minMarkerDistanceRate = 0.05
        aruco_params.minDistanceToBorder = 3
        aruco_params.minOtsuStdDev = 5.0
        aruco_params.perspectiveRemovePixelPerCell = 4
        aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
        aruco_params.maxErroneousBitsInBorderRate = 0.35
        aruco_params.errorCorrectionRate = 0.6
        
        aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, aruco_params)
        corners, ids, rejected = aruco_detector.detectMarkers(gray_enhanced)
        
        # ArUco 마커 검출 결과 확인
        if ids is not None and len(ids) >= 10:  # 최소 10개 마커 필요
            print(f"\rDEBUG: ArUco markers detected: {len(ids)}, proceeding with marker-based pose estimation", end="")
            
            # ArUco 마커의 3D 점들 생성 (보드 좌표계, 미터 단위)
            marker_points_3d = []
            marker_points_2d = []
            
            for i, marker_id in enumerate(ids):
                marker_id = marker_id[0]  # ids는 2D 배열
                
                # 마커의 4개 코너를 3D 점으로 변환
                # 마커의 중심 위치 계산 (보드 좌표계)
                marker_row = marker_id // (self.charuco_squares_x - 1)
                marker_col = marker_id % (self.charuco_squares_x - 1)
                
                # 마커 중심의 3D 좌표 (미터 단위)
                center_x = marker_col * (self.charuco_square_length / 1000.0)  # mm -> m
                center_y = marker_row * (self.charuco_square_length / 1000.0)  # mm -> m
                
                # 마커의 4개 코너를 3D 점으로 생성 (미터 단위)
                half_marker = (self.charuco_marker_length / 1000.0) / 2  # mm -> m
                marker_corners_3d = [
                    [center_x - half_marker, center_y - half_marker, 0],  # 좌상단
                    [center_x + half_marker, center_y - half_marker, 0],  # 우상단
                    [center_x + half_marker, center_y + half_marker, 0],  # 우하단
                    [center_x - half_marker, center_y + half_marker, 0]   # 좌하단
                ]
                
                # 2D 코너 점들
                marker_corners_2d = corners[i][0]  # corners[i]는 4x2 배열
                
                # 3D-2D 매핑에 추가
                marker_points_3d.extend(marker_corners_3d)
                marker_points_2d.extend(marker_corners_2d)
            
            if len(marker_points_3d) >= 12:  # 최소 3개 마커 (3*4=12개 점)
                marker_points_3d = np.array(marker_points_3d, dtype=np.float32)
                marker_points_2d = np.array(marker_points_2d, dtype=np.float32)
                
                # solvePnP로 포즈 추정
                ret, rvec, tvec = cv2.solvePnP(
                    marker_points_3d, marker_points_2d,
                    camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if ret:
                    print(f"\rDEBUG: ArUco marker-based pose estimation successful with {len(ids)} markers", end="")
                    
                    # ArUco 마커 기반 결과를 charuco_corners 형태로 변환
                    # 마커 중심점들을 charuco_corners로 사용
                    charuco_corners_from_markers = []
                    charuco_ids_from_markers = []
                    
                    for i, marker_id in enumerate(ids):
                        marker_id = marker_id[0]  # ids는 2D 배열
                        marker_corners = corners[i][0]  # 4x2 배열
                        
                        # 마커의 중심점 계산
                        center_2d = np.mean(marker_corners, axis=0)
                        charuco_corners_from_markers.append(center_2d)
                        charuco_ids_from_markers.append(marker_id)
                    
                    charuco_corners_from_markers = np.array(charuco_corners_from_markers, dtype=np.float32)
                    charuco_ids_from_markers = np.array(charuco_ids_from_markers, dtype=np.int32)
                    
                    # ArUco 마커 기반 결과를 반환 (charuco_corners 형태로 변환)
                    return rvec, tvec, charuco_corners_from_markers, charuco_ids_from_markers
                else:
                    print(f"\rDEBUG: ArUco marker-based pose estimation failed", end="")
            else:
                print(f"\rDEBUG: Not enough 3D-2D correspondences for ArUco markers ({len(marker_points_3d)} < 12)", end="")
        else:
            print(f"\rDEBUG: Not enough ArUco markers detected ({len(ids) if ids is not None else 0} < 10)", end="")
        
        return None
    
    def get_robot_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        현재 로봇의 포즈를 가져옵니다.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (R_base2ee, t_base2ee) - 미터 단위
        """
        coords = self.robot.get_coords()  # [X, Y, Z, Rx, Ry, Rz] (mm, degree)
        
        # 위치 (mm를 미터로 변환)
        position = np.array(coords[0:3]) / 1000.0  # mm -> m
        
        # 회전 (도를 라디안으로 변환 후 회전 행렬 계산)
        rx, ry, rz = np.radians(coords[3:6])  # degree -> radian
        R_base2ee = self._euler_to_rotation_matrix(rx, ry, rz)
        
        return R_base2ee, position.reshape(3, 1)
    
    def get_robot_angles(self) -> List[float]:
        """
        현재 로봇의 각도를 가져옵니다.
        
        Returns:
            List[float]: 6개 조인트의 각도 (도)
        """
        return self.robot.get_angles()
    
    def evaluate_charuco_quality(self, charuco_corners: np.ndarray, charuco_ids: np.ndarray, 
                                rvec: np.ndarray, tvec: np.ndarray) -> Tuple[bool, str, float]:
        """
        ChArUco 검출 품질을 평가합니다.
        
        Args:
            charuco_corners: 검출된 ChArUco 코너
            charuco_ids: 검출된 ChArUco ID
            rvec: 회전 벡터
            tvec: 변위 벡터
            
        Returns:
            Tuple[bool, str, float]: (품질 통과 여부, 평가 메시지, 품질 점수)
        """
        quality_score = 0.0
        issues = []
        
        # 1. 코너 개수 검사 (안전한 처리)
        corner_count = len(charuco_corners) if charuco_corners is not None else 0
        print(f"\rDEBUG: Evaluating quality - corners: {corner_count}, min_required: {self.min_charuco_corners}", end="")
        
        if charuco_corners is None or corner_count < self.min_charuco_corners:
            issues.append(f"코너 개수 부족 ({corner_count}/{self.min_charuco_corners})")
            quality_score -= 0.1
            print(f"\rDEBUG: Corner count penalty applied, score: {quality_score:.2f}", end="")
        else:
            corner_ratio = corner_count / ((self.charuco_squares_x - 1) * (self.charuco_squares_y - 1))
            quality_score += corner_ratio * 0.6
            print(f"\rDEBUG: Corner ratio: {corner_ratio:.2f}, score: {quality_score:.2f}", end="")
        
        # 2. 거리 검사
        distance = np.linalg.norm(tvec)
        if distance > self.max_distance_threshold:
            issues.append(f"거리 너무 멀음 ({distance*1000:.1f}mm > {self.max_distance_threshold*1000:.1f}mm)")
            quality_score -= 0.2
        elif distance < 0.15:
            issues.append(f"거리 너무 가까움 ({distance*1000:.1f}mm < 150mm)")
            quality_score -= 0.1
        elif distance > 0.6:
            issues.append(f"거리 너무 멀음 ({distance*1000:.1f}mm > 500mm)")
            quality_score -= 0.2
        else:
            distance_score = 1.0 - (distance - 0.15) / (0.5 - 0.15)
            quality_score += distance_score * 0.3
        
        # 3. 회전 각도 검사
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        z_axis = np.array([0, 0, 1])
        camera_z = rotation_matrix @ z_axis
        angle_with_z = np.arccos(np.clip(np.abs(np.dot(camera_z, z_axis)), 0, 1))
        angle_degrees = np.degrees(angle_with_z)
        
        if angle_degrees > 75:
            issues.append(f"각도 너무 극단적 ({angle_degrees:.1f}도)")
            quality_score -= 0.1
        else:
            angle_score = 1.0 - (angle_degrees / 75.0)
            quality_score += angle_score * 0.2
        
        # 4. 코너 분포 검사 (안전한 처리)
        if charuco_corners is not None and corner_count > 0:
            try:
                corners_array = np.array(charuco_corners).reshape(-1, 2)
                std_x = np.std(corners_array[:, 0])
                std_y = np.std(corners_array[:, 1])
                
                if std_x < 30 or std_y < 30:
                    issues.append("코너 분포 집중됨")
                    quality_score -= 0.05
            except Exception as e:
                print(f"\rDEBUG: Corner distribution calculation failed: {e}", end="")
        
        # 5. 기본 점수
        quality_score += 0.3
        print(f"\rDEBUG: Base score added, final score: {quality_score:.2f}", end="")
        
        # 최종 품질 평가
        is_good_quality = quality_score >= self.min_detection_confidence
        
        if is_good_quality:
            message = f"품질 양호 (점수: {quality_score:.2f})"
        else:
            message = f"품질 불량 (점수: {quality_score:.2f}) - {'; '.join(issues)}"
        
        print(f"\rDEBUG: Final quality: {quality_score:.2f}, threshold: {self.min_detection_confidence}, good: {is_good_quality}", end="")
        return is_good_quality, message, quality_score
    
    def _euler_to_rotation_matrix(self, rx: float, ry: float, rz: float) -> np.ndarray:
        """
        오일러 각도(RPY)를 회전 행렬로 변환합니다.
        
        Args:
            rx: X축 회전 (라디안)
            ry: Y축 회전 (라디안)
            rz: Z축 회전 (라디안)
            
        Returns:
            np.ndarray: 3x3 회전 행렬
        """
        # 각 축에 대한 회전 행렬
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
        
        # ZYX 순서로 회전 행렬 곱셈
        return Rz @ Ry @ Rx
    
    def collect_angles_with_camera_feed(self, num_poses: int = 20) -> bool:
        """
        카메라 화면을 보여주면서 실시간 각도 수집 모드 (카메라 캡처 포함)
        
        Args:
            num_poses: 수집할 포즈 개수
            
        Returns:
            bool: 수집 성공 여부
        """
        print(f"\n=== 실시간 각도 수집 모드 (카메라 화면 포함) ===")
        print(f"수집할 포즈 수: {num_poses}")
        print("로봇을 수동으로 조작하여 다양한 포즈의 각도를 수집합니다.")
        print("카메라 화면이 실시간으로 표시되며, ChArUco 검출 상태를 확인할 수 있습니다.")
        print("각 포즈에서 's'를 눌러 저장하거나 'q'를 눌러 종료하세요.")
        print("저장 시 color 이미지, depth, intrinsics도 함께 저장됩니다.")
        
        # RealSense 카메라 초기화 (848x480 해상도)
        config = rs.config()
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)  
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        
        pipeline = None
        try:
            pipeline = rs.pipeline()
            profile = pipeline.start(config)
            print("✅ RealSense 카메라 연결 성공 (848x480)")
        except Exception as e:
            print(f"❌ 카메라 연결 실패: {e}")
        
        # RealSense에서 실제 카메라 내부 파라미터 가져오기
        color_profile = profile.get_stream(rs.stream.color)
        color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        
        # 카메라 행렬 구성
        camera_matrix = np.array([
            [color_intrinsics.fx, 0.0, color_intrinsics.ppx],
            [0.0, color_intrinsics.fy, color_intrinsics.ppy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        # 왜곡 계수
        dist_coeffs = np.array(color_intrinsics.coeffs, dtype=np.float64)
        
        print(f"✅ 실제 카메라 내부 파라미터 로드 완료:")
        print(f"  - 해상도: {color_intrinsics.width}x{color_intrinsics.height}")
        print(f"  - fx: {color_intrinsics.fx:.3f}, fy: {color_intrinsics.fy:.3f}")
        print(f"  - 주점: ({color_intrinsics.ppx:.3f}, {color_intrinsics.ppy:.3f})")
        print(f"  - 왜곡 모델: {color_intrinsics.model}")
        print(f"  - 왜곡 계수: {color_intrinsics.coeffs}")
        
        collected_poses = 0
        pose_index = 0
        
        try:
            while collected_poses < num_poses:
                self.robot.release_all_servos()
                # 카메라 프레임 가져오기
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame:
                    continue
                
                # 이미지 변환
                color_image = np.asanyarray(color_frame.get_data())
                
                # Depth 이미지 변환 (저장용)
                depth_image = None
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
                
                # ChArUco 검출
                charuco_detected = False
                detection_info = "검출 안됨"
                charuco_pose = None
                quality_info = ""
                quality_score = 0.0
                charuco_result = None  # 초기화 추가
                distance = 0.0  # 초기화 추가
                
                try:
                    # 이미지 전처리로 마커 검출 정확도 향상
                    # 1. Grayscale 변환
                    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                    
                    # 2. 대비 향상 (CLAHE - Contrast Limited Adaptive Histogram Equalization)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    gray_enhanced = clahe.apply(gray)
                    
                    # 3. 노이즈 제거 (가우시안 블러) - 제거
                    # gray_enhanced = cv2.GaussianBlur(gray_enhanced, (3, 3), 0)
                    
                    # 4. 이진화 (Otsu's method)
                    _, binary = cv2.threshold(gray_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # 전처리된 이미지를 컬러로 변환 (여기서 미리 생성)
                    gray_enhanced_colored = cv2.cvtColor(gray_enhanced, cv2.COLOR_GRAY2BGR)
                    
                    aruco_params = cv2.aruco.DetectorParameters()
                    # 검출 파라미터 최적화
                    aruco_params.adaptiveThreshWinSizeMin = 3
                    aruco_params.adaptiveThreshWinSizeMax = 23
                    aruco_params.adaptiveThreshWinSizeStep = 10
                    aruco_params.adaptiveThreshConstant = 7
                    aruco_params.minMarkerPerimeterRate = 0.03
                    aruco_params.maxMarkerPerimeterRate = 4.0
                    aruco_params.polygonalApproxAccuracyRate = 0.03
                    aruco_params.minCornerDistanceRate = 0.05
                    aruco_params.minMarkerDistanceRate = 0.05
                    aruco_params.minDistanceToBorder = 3
                    aruco_params.minOtsuStdDev = 5.0
                    aruco_params.perspectiveRemovePixelPerCell = 4
                    aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
                    aruco_params.maxErroneousBitsInBorderRate = 0.35
                    aruco_params.errorCorrectionRate = 0.6
                    
                    aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, aruco_params)
                    corners, ids, rejected = aruco_detector.detectMarkers(gray_enhanced)
                    
                    # 검출된 마커 그리기 (전처리된 이미지에)
                    if ids is not None and len(ids) > 0:
                        cv2.aruco.drawDetectedMarkers(gray_enhanced_colored, corners, ids)
                        # ArUco 마커 개수 저장
                        self._last_marker_count = len(ids)
                        print(f"\rDEBUG: ArUco markers detected: {len(ids)}", end="")
                    else:
                        self._last_marker_count = 0
                    
                    # ChArUco 포즈 검출
                    charuco_result = self.detect_charuco_pose(color_image, camera_matrix, dist_coeffs)
                    if charuco_result is not None:
                        rvec, tvec, charuco_corners, charuco_ids = charuco_result
                        charuco_detected = True
                        distance = np.linalg.norm(tvec)
                        detection_info = f"검출됨 (거리: {distance*1000:.1f}mm)"
                        
                        # 품질 평가
                        if charuco_corners is not None and charuco_ids is not None:
                            print(f"\rDEBUG: ChArUco corners={len(charuco_corners)}, ids={len(charuco_ids)}, min_required={self.min_charuco_corners}", end="")
                            
                            is_good_quality, quality_message, quality_score = self.evaluate_charuco_quality(
                                charuco_corners, charuco_ids, rvec, tvec
                            )
                            quality_info = f"Quality: {quality_score:.2f} ({'GOOD' if is_good_quality else 'LOW'})"
                            
                            # 디버그 정보 출력 (콘솔)
                            print(f"\rChArUco: {len(charuco_corners)} corners, Distance: {distance*1000:.1f}mm, Quality: {quality_score:.2f}", end="")
                            
                            # 품질이 좋은 경우에만 좌표축 그리기 (전처리된 이미지에)
                            if is_good_quality:
                                cv2.drawFrameAxes(gray_enhanced_colored, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
                        else:
                            # ArUco 마커만 검출된 경우에도 품질 평가 시도
                            print(f"\rDEBUG: ArUco markers only, trying alternative quality assessment", end="")
                            
                            # ArUco 마커 개수를 기반으로 정교한 품질 평가
                            if hasattr(self, '_last_marker_count'):
                                marker_count = self._last_marker_count
                                if marker_count >= 15:  # 충분한 ArUco 마커가 검출된 경우
                                    quality_score = 0.5  # 더 높은 점수
                                    is_good_quality = quality_score >= self.min_detection_confidence
                                    quality_info = f"Quality: {quality_score:.2f} (ArUco: {marker_count}) ({'GOOD' if is_good_quality else 'LOW'})"
                                    print(f"\rArUco: {marker_count} markers, Quality: {quality_score:.2f}", end="")
                                elif marker_count >= 10:  # 중간 수준의 ArUco 마커
                                    quality_score = 0.3
                                    is_good_quality = quality_score >= self.min_detection_confidence
                                    quality_info = f"Quality: {quality_score:.2f} (ArUco: {marker_count}) ({'GOOD' if is_good_quality else 'LOW'})"
                                    print(f"\rArUco: {marker_count} markers, Quality: {quality_score:.2f}", end="")
                                else:
                                    quality_score = 0.1
                                    is_good_quality = False
                                    quality_info = f"Quality: {quality_score:.2f} (ArUco: {marker_count}) (LOW)"
                                    print(f"\rArUco: {marker_count} markers (insufficient)", end="")
                            else:
                                quality_score = 0.0
                                is_good_quality = False
                                quality_info = "Quality: N/A (ArUco only)"
                                print(f"\rChArUco: ArUco markers only", end="")
                        
                except Exception as e:
                    detection_info = f"검출 오류: {str(e)[:20]}"
                    # 오류 발생 시 기본 이미지 사용
                    gray_enhanced_colored = color_image.copy()
                
                # 현재 로봇 상태 가져오기
                try:
                    angles = self.get_robot_angles()
                    coords = self.robot.get_coords()
                except Exception as e:
                    angles = [0, 0, 0, 0, 0, 0]
                    coords = [0, 0, 0, 0, 0, 0]
                    # 실시간 모드에서는 오류 메시지 출력하지 않음
                
                # 화면에 정보 표시 (전처리된 이미지 사용)
                # gray_enhanced_colored는 이미 위에서 생성됨
                info_image = gray_enhanced_colored.copy()
                
                # 상태 정보 텍스트 (영어로 표시)
                cv2.putText(info_image, f"Pose: {collected_poses}/{num_poses}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 전처리 정보 표시
                cv2.putText(info_image, "Enhanced Image (CLAHE only)", (10, 225), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # ChArUco 코너 검출 상태 표시
                if charuco_result is not None:
                    rvec, tvec, charuco_corners, charuco_ids = charuco_result
                    corner_count = len(charuco_corners) if charuco_corners is not None else 0
                    id_count = len(charuco_ids) if charuco_ids is not None else 0
                    cv2.putText(info_image, f"ChArUco Corners: {corner_count}, IDs: {id_count}", (10, 205), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                else:
                    cv2.putText(info_image, "ChArUco Corners: NOT DETECTED", (10, 205), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                # ChArUco 검출 상태 표시
                if charuco_detected:
                    cv2.putText(info_image, f"ChArUco: DETECTED ({distance*1000:.1f}mm)", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(info_image, "ChArUco: NOT DETECTED", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # 품질 정보 표시
                if quality_info:
                    quality_color = (0, 255, 0) if quality_score >= self.min_detection_confidence else (0, 0, 255)
                    cv2.putText(info_image, f"Quality: {quality_score:.2f} ({'GOOD' if quality_score >= self.min_detection_confidence else 'LOW'})", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 1)
                
                cv2.putText(info_image, f"Angles: [{angles[0]:.1f}, {angles[1]:.1f}, {angles[2]:.1f}]", (10, 135),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(info_image, f"        [{angles[3]:.1f}, {angles[4]:.1f}, {angles[5]:.1f}]", (10, 155), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(info_image, f"Coords: [{coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}]", (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 저장 가능 여부 표시
                if charuco_detected:
                    corner_count = len(charuco_corners) if charuco_corners is not None else 0
                    if corner_count >= 6:
                        cv2.putText(info_image, "Press 's' to save", (10, 250), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.putText(info_image, f"Need >=6 corners (current: {corner_count})", (10, 250), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                else:
                    cv2.putText(info_image, "No ChArUco detected - cannot save", (10, 250), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # 화면 표시
                cv2.imshow('Eye-to-Hand Calibration - Real-time', info_image)
                
                # 키보드 입력 확인
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print(f"\n 사용자에 의해 중단되었습니다. 수집된 포즈: {collected_poses}")
                    break
                elif key == ord('p'):
                    # 로봇 고정 및 안정화 대기
                    print(f"\n🔒 로봇을 고정하고 안정화 대기 중... (1초)")
                    self.robot.power_on()
                    time.sleep(1.0)  # 1초 대기로 로봇 안정화
                    
                    # 안정화 후 다시 카메라 프레임과 ChArUco 검출
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        color_image = np.asanyarray(color_frame.get_data())
                        charuco_result = self.detect_charuco_pose(color_image, camera_matrix, dist_coeffs)
                        
                        if charuco_result is not None:
                            rvec, tvec, charuco_corners, charuco_ids = charuco_result
                            distance = np.linalg.norm(tvec)
                            
                            # 최소 코너 개수 체크 (6개 이상 필요)
                            corner_count = len(charuco_corners) if charuco_corners is not None else 0
                            if corner_count < 6:
                                print(f"\n⚠️ 저장 건너뜀: 코너 개수 부족 ({corner_count} < 6)")
                                print("ChArUco 보드가 더 잘 보이도록 조정해주세요.")
                                continue
                            
                            # 품질 평가
                            if charuco_corners is not None and charuco_ids is not None:
                                is_good_quality, quality_message, quality_score = self.evaluate_charuco_quality(
                                    charuco_corners, charuco_ids, rvec, tvec
                                )
                            else:
                                # ArUco 마커 기반 fallback 결과인 경우
                                is_good_quality = True  # ArUco 마커 기반은 기본적으로 양호로 간주
                                quality_score = 0.5
                            
                            if is_good_quality:
                                # 현재 로봇 상태 가져오기
                                angles = self.get_robot_angles()
                                coords = self.robot.get_coords()
                                
                                # 데이터 저장
                                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                                
                                # 각도 데이터 저장
                                angle_data = {
                                    "angles": angles,
                                    "coords": coords,
                                    "pose_index": pose_index,
                                    "timestamp": timestamp,
                                    "stabilization_wait": True
                                }
                                angle_path = self.angles_dir / f"{pose_index:02d}_{timestamp}_angles.json"
                                with open(angle_path, 'w') as f:
                                    json.dump(angle_data, f, indent=2)
                                
                                # 현재 로봇 포즈 가져오기
                                R_base2ee, t_base2ee = self.get_robot_pose()
                                
                                # 로봇 포즈 저장
                                pose_data = {
                                    "R_base2ee": R_base2ee.tolist(),
                                    "t_base2ee": t_base2ee.tolist(),
                                    "angles": angles,
                                    "pose_index": pose_index,
                                    "timestamp": timestamp
                                }
                                pose_path = self.poses_dir / f"{pose_index:02d}_{timestamp}_pose.json"
                                with open(pose_path, 'w') as f:
                                    json.dump(pose_data, f, indent=2)
                                
                                # Color 이미지 저장
                                color_path = self.save_dir / "color" / f"{pose_index:02d}_{timestamp}.jpg"
                                color_path.parent.mkdir(parents=True, exist_ok=True)
                                cv2.imwrite(str(color_path), color_image)
                                
                                # Depth 이미지 저장 (numpy 배열로)
                                if depth_image is not None:
                                    depth_path = self.save_dir / "depth" / f"{pose_index:02d}_{timestamp}.npy"
                                    depth_path.parent.mkdir(parents=True, exist_ok=True)
                                    np.save(str(depth_path), depth_image)
                                    
                                    # Depth 이미지를 PNG로도 저장 (시각화용)
                                    depth_png_path = self.save_dir / "depth" / "converted_png" / f"{pose_index:02d}_{timestamp}.png"
                                    depth_png_path.parent.mkdir(parents=True, exist_ok=True)
                                    # Depth 이미지를 0-255 범위로 정규화
                                    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                                    cv2.imwrite(str(depth_png_path), depth_normalized)
                                
                                # Intrinsics 저장 (실제 카메라 파라미터 사용)
                                intrinsics_data = {
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
                                    "camera_matrix": camera_matrix.tolist(),
                                    "dist_coeffs": dist_coeffs.tolist(),
                                    "timestamp": datetime.now().isoformat()
                                }
                                intrinsics_path = self.save_dir / "intrinsics" / f"{pose_index:02d}_{timestamp}.json"
                                intrinsics_path.parent.mkdir(parents=True, exist_ok=True)
                                with open(intrinsics_path, 'w') as f:
                                    json.dump(intrinsics_data, f, indent=2)
                                
                                # ChArUco 데이터 저장 (타겟 -> 카메라) - Eye-to-Hand 구조
                                # detect_charuco_pose()는 카메라 -> 타겟을 반환하므로 역행렬을 취함
                                R_cam2target, _ = cv2.Rodrigues(rvec)
                                t_cam2target = tvec
                                
                                # 타겟 -> 카메라 변환 (역행렬)
                                R_target2cam = R_cam2target.T
                                t_target2cam = -R_target2cam @ t_cam2target
                                
                                charuco_data = {
                                    "rvec_target2cam": cv2.Rodrigues(R_target2cam)[0].tolist(),
                                    "tvec_target2cam": t_target2cam.tolist(),
                                    "rvec_cam2target": rvec.tolist(),  # 원본 데이터도 보존
                                    "tvec_cam2target": tvec.tolist(),
                                    "pose_index": pose_index,
                                    "timestamp": timestamp,
                                    "charuco_corners_count": len(charuco_corners) if charuco_corners is not None else 0,
                                    "charuco_ids_count": len(charuco_ids) if charuco_ids is not None else 0,
                                    "distance_mm": float(distance * 1000),
                                    "quality_score": float(quality_score),
                                    "detection_method": "charuco_corners" if charuco_corners is not None else "aruco_markers",
                                    "color_image_path": str(color_path),
                                    "depth_image_path": str(depth_path) if depth_image is not None else None,
                                    "intrinsics_path": str(intrinsics_path)
                                }
                                charuco_path = self.save_dir / f"{pose_index:02d}_{timestamp}_charuco.json"
                                with open(charuco_path, 'w') as f:
                                    json.dump(charuco_data, f, indent=2)
                                
                                print(f"\n✅ 포즈 {pose_index} 저장 완료!")
                                print(f"  - 거리: {distance*1000:.1f}mm")
                                print(f"  - 품질: {quality_score:.2f}")
                                print(f"  - 검출 방법: {'ChArUco 코너' if charuco_corners is not None else 'ArUco 마커'}")
                                print(f"  - 각도 데이터: {angle_path}")
                                print(f"  - 로봇 포즈: {pose_path}")
                                print(f"  - ChArUco 포즈: {charuco_path}")
                                print(f"  - Color 이미지: {color_path}")
                                if depth_image is not None:
                                    print(f"  - Depth 이미지: {depth_path}")
                                    print(f"  - Depth PNG: {depth_png_path}")
                                print(f"  - Intrinsics: {intrinsics_path}")
                                
                                collected_poses += 1
                                pose_index += 1
                            else:
                                print(f"\n❌ 품질이 낮아 저장하지 않습니다. (품질 점수: {quality_score:.2f})")
                        else:
                            print(f"\n❌ ChArUco 검출 실패로 저장하지 않습니다.")
                    
        except KeyboardInterrupt:
            print(f"\n 사용자에 의해 중단되었습니다. 수집된 포즈: {collected_poses}")
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
        finally:
            # 카메라 정리
            if pipeline is not None:
                try:
                    pipeline.stop()
                except:
                    pass
            cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print(f" 실시간 각도 수집 완료")
        print(f" 총 {collected_poses}개의 포즈 데이터가 수집되었습니다.")
        print(f"{'='*60}")
        
        return collected_poses >= 10  # 최소 10개 이상 필요

    def calculate_transformation_matrix(self) -> bool:
        """
        수집된 데이터를 사용하여 Eye-to-Hand 변환 행렬을 계산합니다.
        """
        print("\n=== Eye-to-Hand 변환 행렬 계산 중 ===")
        
        try:
            # 저장된 데이터 로드
            pose_files = list(self.poses_dir.glob("*_pose.json"))
            charuco_files = list(self.save_dir.glob("*_charuco.json"))
            
            if len(charuco_files) < 3:
                print(f"충분한 데이터가 없습니다. (필요: 3개 이상, 현재: {len(charuco_files)}개)")
                return False
            
            # 데이터 정렬 (인덱스 순서대로)
            pose_files.sort(key=lambda x: int(x.stem.split('_')[0]))
            charuco_files.sort(key=lambda x: int(x.stem.split('_')[0]))
            
            T_base2ee_list = []  # 로봇 베이스 -> End-Effector (미터 단위)
            T_cam2target_list = []  # 카메라 -> ChArUco 타겟 (미터 단위)
            quality_weights = []  # 품질 점수를 가중치로 사용
            
            print(f"총 {len(charuco_files)}개의 포즈 데이터를 로드합니다...")
            
            # 모든 charuco 파일을 사용 (ChArUco 코너 기반과 ArUco 마커 기반 모두 포함)
            valid_pose_files = []
            valid_charuco_files = []
            
            for charuco_file in charuco_files:
                # charuco 파일이 존재하면 사용
                valid_charuco_files.append(charuco_file)
                # 해당하는 pose 파일 찾기
                pose_index = int(charuco_file.stem.split('_')[0])
                pose_file = self.poses_dir / f"{pose_index:02d}_*_pose.json"
                pose_files_found = list(self.poses_dir.glob(f"{pose_index:02d}_*_pose.json"))
                if pose_files_found:
                    valid_pose_files.append(pose_files_found[0])
                else:
                    # pose 파일이 없으면 angles 파일에서 각도 정보 사용
                    angle_files_found = list(self.angles_dir.glob(f"{pose_index:02d}_*_angles.json"))
                    if angle_files_found:
                        with open(angle_files_found[0], 'r') as f:
                            angle_data = json.load(f)
                        # 각도에서 pose 계산 (임시 pose 파일 생성)
                        angles = angle_data["angles"]
                        coords = angle_data["coords"]
                        # 여기서 각도를 pose로 변환하는 로직 추가 필요
                        valid_pose_files.append(None)  # 임시로 None 추가
            
            print(f"유효한 포즈: {len(valid_charuco_files)}개")
            
            if len(valid_charuco_files) < 3:
                print(f"유효한 포즈가 부족합니다. (필요: 3개 이상, 현재: {len(valid_charuco_files)}개)")
                return False
            
            for i, charuco_file in enumerate(valid_charuco_files):
                # ChArUco 포즈 로드 (타겟 -> 카메라) - Eye-to-Hand 구조
                with open(charuco_file, 'r') as f:
                    charuco_data = json.load(f)
                
                # 품질 점수를 가중치로 사용
                quality_score = charuco_data.get("quality_score", 0.5)  # 기본값 0.5
                quality_weights.append(quality_score)
                
                # 새로운 형식에서는 target2cam 데이터를 사용
                if "rvec_target2cam" in charuco_data:
                    rvec = np.array(charuco_data["rvec_target2cam"])
                    tvec = np.array(charuco_data["tvec_target2cam"])
                else:
                    # 기존 형식 호환성을 위해 역행렬 계산
                    rvec_old = np.array(charuco_data["rvec"])
                    tvec_old = np.array(charuco_data["tvec"])
                    R_cam2target, _ = cv2.Rodrigues(rvec_old)
                    R_target2cam = R_cam2target.T
                    t_target2cam = -R_target2cam @ tvec_old
                    rvec = cv2.Rodrigues(R_target2cam)[0]
                    tvec = t_target2cam
                
                # 회전 벡터를 회전 행렬로 변환
                R_cam2target, _ = cv2.Rodrigues(rvec)
                
                # 카메라 -> 타겟 (미터 단위) - Eye-to-Hand 구조에 맞게 수정
                T_cam2target = np.eye(4)
                T_cam2target[:3, :3] = R_cam2target.T  # 역행렬
                T_cam2target[:3, 3] = -R_cam2target.T @ tvec.flatten()  # 역변환
                T_cam2target_list.append(T_cam2target)
                
                # 로봇 포즈 처리 (pose 파일이 있는 경우)
                if i < len(valid_pose_files) and valid_pose_files[i] is not None:
                    with open(valid_pose_files[i], 'r') as f:
                        pose_data = json.load(f)
                    
                    t_base2ee = np.array(pose_data["t_base2ee"])
                    R_base2ee = np.array(pose_data["R_base2ee"])
                else:
                    # angles 파일에서 pose 계산
                    angle_files_found = list(self.angles_dir.glob(f"{charuco_file.stem.split('_')[0]}_*_angles.json"))
                    if angle_files_found:
                        with open(angle_files_found[0], 'r') as f:
                            angle_data = json.load(f)
                        
                        coords = angle_data["coords"]
                        # coords에서 pose 계산 (간단한 변환)
                        position = np.array(coords[0:3]) / 1000.0  # mm를 m로 변환
                        rx, ry, rz = np.radians(coords[3:6])
                        R_base2ee = self._euler_to_rotation_matrix(rx, ry, rz)
                        t_base2ee = position.reshape(3, 1)
                    else:
                        print(f"포즈 {i}에 대한 각도 데이터를 찾을 수 없습니다.")
                        continue
                
                # 로봇 베이스 -> End-Effector (미터 단위)
                T_base2ee = np.eye(4)
                T_base2ee[:3, :3] = R_base2ee
                T_base2ee[:3, 3] = t_base2ee.flatten()
                T_base2ee_list.append(T_base2ee)
                
                # 디버그 정보
                detection_method = charuco_data.get("detection_method", "unknown")
                distance_mm = charuco_data.get("distance_mm", 0)
                print(f"포즈 {i+1}: {detection_method}, 거리={distance_mm:.1f}mm, 품질={quality_score:.3f}")
            
            # 품질 점수 기반 가중치 정규화
            quality_weights = np.array(quality_weights)
            quality_weights = quality_weights / np.sum(quality_weights)  # 합이 1이 되도록 정규화
            
            print(f"\n품질 점수 기반 가중치:")
            for i, (weight, score) in enumerate(zip(quality_weights, quality_weights * np.sum(quality_weights))):
                print(f"  포즈 {i+1}: 품질={score:.3f}, 가중치={weight:.3f}")
            
            # Eye-to-Hand calibration 수행 (가중치 포함)
            print("\nEye-to-Hand calibration 계산 중...")
            R_cam2base, t_cam2base = self._solve_eye_to_hand_calibration(
                T_cam2target_list, T_base2ee_list, quality_weights
            )
            
            if R_cam2base is not None:
                # 결과는 미터 단위로 반환됨
                t_cam2base_norm = np.linalg.norm(t_cam2base)
                print(f"t_cam2base 크기 (m): {t_cam2base_norm:.3f}")
                
                # 미터 단위로 저장 (표준 단위)
                print(f"단위: 미터 (표준 단위)")
                
                # 변환 행렬 생성
                T_cam2base = np.eye(4)
                T_cam2base[:3, :3] = R_cam2base
                T_cam2base[:3, 3] = t_cam2base.flatten()
                
                # 결과 저장 (미터 단위)
                calibration_result = {
                    "R_cam2base": R_cam2base.tolist(),
                    "t_cam2base": t_cam2base.flatten().tolist(),  # 미터 단위로 저장
                    "num_poses": len(T_base2ee_list),
                    "timestamp": datetime.now().isoformat(),
                    "description": "카메라 -> 로봇 베이스 변환 행렬 (Eye-to-Hand calibration, 미터 단위, 품질 가중치 적용)",
                    "coordinate_transformation": {
                        "applied": False,
                        "description": "축보정 미적용 - 원본 카메라 좌표계 유지"
                    },
                    "quality_weighting": {
                        "applied": True,
                        "description": "품질 점수 기반 가중치 적용",
                        "weights": quality_weights.tolist(),
                        "quality_scores": (quality_weights * np.sum(quality_weights)).tolist()
                    }
                }
                
                result_path = self.save_dir / "cam2base.json"
                with open(result_path, 'w') as f:
                    json.dump(calibration_result, f, indent=2)
                
                # numpy 배열을 안전하게 float로 변환
                t_x = float(t_cam2base.flatten()[0])
                t_y = float(t_cam2base.flatten()[1])
                t_z = float(t_cam2base.flatten()[2])
                
                print(f"✅ Eye-to-Hand 캘리브레이션 완료!")
                print(f"결과 저장: {result_path}")
                print(f"카메라 -> 로봇 베이스 변환 행렬:")
                print(f"R:\n{R_cam2base}")
                print(f"t (m): [{t_x:.3f}, {t_y:.3f}, {t_z:.3f}]")
                print(f"품질 가중치 적용됨: {len(quality_weights)}개 포즈")
                
                return True
            else:
                print("❌ 변환 행렬 계산 실패")
                return False
                
        except Exception as e:
            print(f"캘리브레이션 계산 중 오류: {e}")
            return False
    
    def _solve_eye_to_hand_calibration(self, T_cam2target_list, T_base2ee_list, weights=None):
        """
        Eye-to-Hand calibration 방정식을 해결합니다.
        
        논리 구조:
        T_base2target = T_base2ee (ee와 target이 고정 관계)
        T_base2target = T_base2cam * T_cam2target
        T_base2cam = T_base2target * T_cam2target.inverse()
        T_cam2base = T_cam2target * T_base2ee.inverse()
        
        Args:
            T_cam2target_list: 카메라 -> 타겟 변환 리스트 (미터 단위)
            T_base2ee_list: 베이스 -> End-Effector 변환 리스트 (미터 단위)
            weights: 품질 점수 기반 가중치 리스트 (None이면 균등 가중치)
            
        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: (R_cam2base, t_cam2base) - 미터 단위
        """
        try:
            # 충분한 데이터가 있는지 확인
            if len(T_cam2target_list) < 3:
                print("충분한 포즈 데이터가 없습니다.")
                return None, None
            
            print(f"Eye-to-Hand calibration 계산 중... (총 {len(T_cam2target_list)}개 포즈)")
            
            # 각 포즈에서 T_cam2base 계산
            R_cam2base_list = []
            t_cam2base_list = []
            
            for i, (T_cam2target, T_base2ee) in enumerate(zip(T_cam2target_list, T_base2ee_list)):
                try:
                    # T_cam2base = T_cam2target * T_base2ee.inverse()
                    T_base2ee_inv = np.linalg.inv(T_base2ee)
                    T_cam2base = T_cam2target @ T_base2ee_inv
                    
                    R_cam2base = T_cam2base[:3, :3]
                    t_cam2base = T_cam2base[:3, 3]
                    
                    R_cam2base_list.append(R_cam2base)
                    t_cam2base_list.append(t_cam2base)
                    
                    print(f"포즈 {i+1}: T_cam2base 계산 완료")
                    
                except Exception as e:
                    print(f"포즈 {i+1} 계산 실패: {e}")
                    continue
            
            if len(R_cam2base_list) < 3:
                print("유효한 계산 결과가 부족합니다.")
                return None, None
            
            # 가중치 처리
            if weights is None or len(weights) != len(R_cam2base_list):
                # 균등 가중치
                weights = np.ones(len(R_cam2base_list)) / len(R_cam2base_list)
                print("균등 가중치 적용")
            else:
                # 가중치 정규화 (합이 1이 되도록)
                weights = np.array(weights, dtype=float)
                weights = weights / np.sum(weights)
                print(f"품질 점수 기반 가중치 적용 (총 {len(weights)}개)")
            
            # 1. 평행이동 가중평균 (직접 계산)
            t_cam2base_avg = np.zeros(3)
            for i, (t_cam2base, weight) in enumerate(zip(t_cam2base_list, weights)):
                t_cam2base_avg += weight * t_cam2base
                print(f"  포즈 {i+1}: t 가중치={weight:.3f}, t={t_cam2base}")
            
            print(f"가중평균 t_cam2base: {t_cam2base_avg}")
            
            # 2. 회전 행렬 쿼터니언 가중평균
            print("쿼터니언 가중평균으로 회전 행렬 계산 중...")
            
            # 각 회전 행렬을 쿼터니언으로 변환
            quaternions = []
            for i, R in enumerate(R_cam2base_list):
                try:
                    # 회전 행렬을 쿼터니언으로 변환
                    q = self._rotation_matrix_to_quaternion(R)
                    quaternions.append(q)
                    print(f"  포즈 {i+1}: 쿼터니언 변환 완료")
                except Exception as e:
                    print(f"  포즈 {i+1}: 쿼터니언 변환 실패 - {e}")
                    # 실패한 경우 단위 쿼터니언 사용
                    quaternions.append(np.array([1.0, 0.0, 0.0, 0.0]))
            
            # 쿼터니언 가중평균 계산
            if len(quaternions) > 0:
                R_cam2base_final = self._weighted_average_quaternions(quaternions, weights)
                print("쿼터니언 가중평균 완료")
            else:
                print("쿼터니언 변환 실패로 기존 방식 사용")
                # 기존 방식 (요소별 가중평균 후 SVD 정규화)
                R_cam2base_avg = np.zeros((3, 3))
                for i, (R, weight) in enumerate(zip(R_cam2base_list, weights)):
                    R_cam2base_avg += weight * R
                
                # 회전 행렬을 직교 행렬로 정규화
                U, _, Vt = np.linalg.svd(R_cam2base_avg)
                R_cam2base_final = U @ Vt
                print("기존 방식 (요소별 가중평균 + SVD) 사용")
            
            print(f"최종 R_cam2base:\n{R_cam2base_final}")
            print(f"최종 t_cam2base: {t_cam2base_avg}")
            
            return R_cam2base_final, t_cam2base_avg
            
        except Exception as e:
            print(f"Eye-to-Hand calibration 계산 오류: {e}")
            return None, None
    
    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """
        회전 행렬을 쿼터니언으로 변환합니다.
        
        Args:
            R: 3x3 회전 행렬
            
        Returns:
            np.ndarray: [w, x, y, z] 형태의 쿼터니언
        """
        # 회전 행렬의 대각선 요소들의 합
        trace = np.trace(R)
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
        
        return np.array([w, x, y, z])
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        쿼터니언을 회전 행렬로 변환합니다.
        
        Args:
            q: [w, x, y, z] 형태의 쿼터니언
            
        Returns:
            np.ndarray: 3x3 회전 행렬
        """
        w, x, y, z = q
        
        # 쿼터니언 정규화
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm > 0:
            w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # 회전 행렬 계산
        R = np.array([
            [1-2*y*y-2*z*z,     2*x*y-2*w*z,     2*x*z+2*w*y],
            [    2*x*y+2*w*z, 1-2*x*x-2*z*z,     2*y*z-2*w*x],
            [    2*x*z-2*w*y,     2*y*z+2*w*x, 1-2*x*x-2*y*y]
        ])
        
        return R
    
    def _weighted_average_quaternions(self, quaternions: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
        """
        쿼터니언들의 가중평균을 계산합니다.
        
        Args:
            quaternions: 쿼터니언 리스트
            weights: 가중치 배열 (합이 1)
            
        Returns:
            np.ndarray: 가중평균 쿼터니언
        """
        if len(quaternions) == 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        if len(quaternions) == 1:
            return quaternions[0]
        
        # 가중합 계산 (쿼터니언의 부호를 맞춤)
        weighted_sum = np.zeros(4)
        
        for i, (q, weight) in enumerate(zip(quaternions, weights)):
            # 첫 번째 쿼터니언과의 내적을 확인하여 부호 결정
            if i == 0:
                weighted_sum += weight * q
            else:
                # 첫 번째 쿼터니언과의 내적이 음수면 부호 반전
                dot_product = np.dot(quaternions[0], q)
                if dot_product < 0:
                    weighted_sum += weight * (-q)
                else:
                    weighted_sum += weight * q
        
        # 정규화
        norm = np.linalg.norm(weighted_sum)
        if norm > 0:
            weighted_sum = weighted_sum / norm
        
        # 쿼터니언을 회전 행렬로 변환
        R_final = self._quaternion_to_rotation_matrix(weighted_sum)
        
        return R_final



def main():
    """
    메인 함수
    """
    print("Eye-to-Hand Calibration for MyCobot280")
    print("=" * 50)
    print("✨ 새로운 기능: 품질 점수 기반 가중치 적용")
    print("   - 각 포즈의 quality_score를 가중치로 사용")
    print("   - 회전: 쿼터니언 가중평균으로 더 정확한 결과")
    print("   - 평행이동: 가중평균으로 잡음에 강건한 결과")
    print("🔒 안전 기능: 최소 6개 코너 검출 시에만 포즈 추정")
    print("   - OpenCV 오류 방지 및 안정적인 캘리브레이션")
    print("=" * 50)
    
    # 캘리브레이터 초기화 (실제 보드 크기에 맞춤)
    calibrator = EyeToHandCalibrator(
        robot_port="/dev/ttyACM0",
        robot_baud=115200,
        save_dir="/home/ros/llm_robot/data/Calibration/Eye-to-Hand14",
        charuco_squares_x=7, 
        charuco_squares_y=5,   
        charuco_square_length=28.0, # mm (실제 측정값)
        charuco_marker_length=14.0, # mm (실제 측정값)
        min_charuco_corners=10,  # 최소 ChArUco 코너 개수 (더 완화)
        min_detection_confidence=0.2,  # 최소 검출 신뢰도 (더욱 완화)
        max_distance_threshold=0.7)  # 최대 거리 임계값 (미터)
    
    
    try:
        print("\n=== 캘리브레이션 옵션 ===")
        print("1. 실시간 각도 수집 모드 (카메라 화면 포함)")
        print("2. 종료")
        
        choice = input("선택 (1/2): ").strip()
        
        if choice == "1":
            # 실시간 각도 수집 모드 (카메라 화면 포함)
            num_poses = int(input("\n수집할 포즈 수를 입력하세요 (권장: 25-30): ") or "25")
            print(f"\n{num_poses}개의 포즈로 실시간 각도 수집을 시작합니다...")
            print("카메라 화면이 표시되며, ChArUco 검출 상태를 실시간으로 확인할 수 있습니다.")
            print("각 포즈에서 's'를 눌러 저장하거나 'q'를 눌러 종료하세요.")
            
            collection_success = calibrator.collect_angles_with_camera_feed(num_poses)
            
            if collection_success:
                print("\n🎉 실시간 각도 수집 완료!")
                print("\n📊 수집된 데이터로 cam2ee.json을 자동 계산합니다...")
                calibration_success = calibrator.calculate_transformation_matrix()
                if calibration_success:
                    print("\n🎉 Eye-to-Hand 캘리브레이션 완료!")
                else:
                    print("\n❌ 캘리브레이션 실패")
            else:
                print("\n❌ 각도 수집 실패")
        else:
            print("\n종료합니다.")
            
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()