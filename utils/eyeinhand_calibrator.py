#!/usr/bin/env python3
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


class CamToGripperCalibrator:
    """
    Cam-to-Gripper calibration을 수행하는 클래스
    
    ChArUco 보드를 End-Effector에 고정하고 다양한 포즈에서 이미지를 캡처하여
    카메라와 로봇 Gripper 간의 변환 행렬을 계산합니다.
    """
    
    def __init__(self, 
                 robot_port: str = "/dev/ttyACM0",
                 robot_baud: int = 115200,
                 save_dir: str = "/home/ros/llm_robot/data/Calibration/Cam-to-Gripper",
                 charuco_squares_x: int = 9,   # 가로 사각형 개수 (columns)
                 charuco_squares_y: int = 6,   # 세로 사각형 개수 (rows)
                 charuco_square_length: float = 28.0,  # mm
                 charuco_marker_length: float = 21.0,  # mm
                 min_charuco_corners: int = 25,  # 최소 ChArUco 코너 개수
                 min_detection_confidence: float = 0.5,  # 최소 검출 신뢰도
                 max_distance_threshold: float = 0.7):  # 최대 거리 임계값 (미터)
        """
        Cam-to-Gripper calibration 초기화
        
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
        
        print(f"Cam-to-Gripper Calibrator 초기화 완료")
        print(f"ChArUco 크기: {charuco_squares_x}x{charuco_squares_y}")
        print(f"사각형 길이: {charuco_square_length:.1f}mm")
        print(f"마커 길이: {charuco_marker_length:.1f}mm")
        print(f"Cam-to-Gripper calibration 모드")
        print(f"이상치 처리 파라미터:")
        print(f"  - 최소 ChArUco 코너: {min_charuco_corners}개")
        print(f"  - 최소 검출 신뢰도: {min_detection_confidence}")
        print(f"  - 최대 거리 임계값: {max_distance_threshold}m")
        print(f"좌표축 변환:")
        print(f"  - 카메라 Z축 → 로봇 X축")
        print(f"  - 카메라 -X축 → 로봇 Y축")
        print(f"  - 카메라 -Y축 → 로봇 Z축")
    
    def _load_camera_intrinsics(self, intrinsics_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        카메라 내부 파라미터를 로드합니다.
        
        Args:
            intrinsics_path: 카메라 내부 파라미터 JSON 파일 경로
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (camera_matrix, dist_coeffs)
        """
        try:
            camera_matrix, dist_coeffs = load_intrinsics(intrinsics_path)
            print(f"카메라 내부 파라미터 로드 완료: {intrinsics_path}")
            return camera_matrix, dist_coeffs
        except Exception as e:
            print(f"카메라 내부 파라미터 로드 실패: {e}")
            # 기본 카메라 행렬 사용 (RealSense D455 848x480 해상도용)
            # 카메라 내부 파라미터 가져오기 (1280x800 해상도용)
            camera_matrix = np.array([
                [642.947,   0.0,     644.808],
                [0.0,       642.073, 409.050],
                [0.0,       0.0,     1.0]
            ], dtype=np.float64)
            
            # 왜곡 계수 (k1, k2, p1, p2, k3)
            dist_coeffs = np.array([
                -0.05594644322991371,
                0.06878077983856201,
                -0.00011232726683374494,
                0.000743341923225671,
                -0.022005939856171608
            ], dtype=np.float64)
            return camera_matrix, dist_coeffs
    


    def detect_charuco_pose(self, image: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        이미지에서 ChArUco 보드의 포즈를 검출합니다.
        ChArUco 코너 기반 pose 추정 실패 시 ArUco 마커 기반 fallback 전략 사용
        """
        # 원본 이미지
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 전처리된 이미지 (대비 향상)
        # 1. CLAHE로 대비 향상 (clipLimit 증가)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        gray_enhanced = clahe.apply(gray)
        
        # 보드 설정 정보 출력
        print(f"\rDEBUG: Board config - size: ({self.charuco_squares_x}, {self.charuco_squares_y}), square_length: {self.charuco_square_length*1000:.1f}mm, marker_length: {self.charuco_marker_length*1000:.1f}mm", end="")
        
        # 단위 변환: mm -> meter (함수 시작 부분에서 정의)
        square_length_m = self.charuco_square_length * 0.001  # mm -> m
        marker_length_m = self.charuco_marker_length * 0.001  # mm -> m
        
        # 1. ChArUco 코너 기반 pose 추정 시도
        charuco_params = cv2.aruco.CharucoParameters()
        charuco_params.minMarkers = 1
        charuco_params.tryRefineMarkers = True
        
        charuco_detector = cv2.aruco.CharucoDetector(self.charuco_board, charuco_params)
        
        # 원본 이미지로 시도
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
        print(f"\rDEBUG: Original image - ChArUco corners: {len(charuco_corners) if charuco_corners is not None else 0}, IDs: {len(charuco_ids) if charuco_ids is not None else 0}, markers: {len(marker_corners) if marker_corners is not None else 0}", end="")
        
        # 전처리된 이미지로 시도
        if charuco_corners is None or len(charuco_corners) == 0:
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray_enhanced)
            print(f"\rDEBUG: Enhanced image - ChArUco corners: {len(charuco_corners) if charuco_corners is not None else 0}, IDs: {len(charuco_ids) if charuco_ids is not None else 0}, markers: {len(marker_corners) if marker_corners is not None else 0}", end="")
        
        # ChArUco 코너 기반 pose 추정 성공 시
        if (charuco_corners is not None and charuco_ids is not None and
            len(charuco_corners) > 2 and len(charuco_corners) == len(charuco_ids)):
            
            print(f"\rDEBUG: ChArUco corners detected: {len(charuco_corners)}, proceeding with corner-based pose estimation", end="")
            
            # ChArUco 보드의 3D 점들 생성 (보드 좌표계, 미터 단위)
            board_points_3d = []
            for i in range(self.charuco_squares_y - 1):
                for j in range(self.charuco_squares_x - 1):
                    x = j * square_length_m  # 미터 단위
                    y = i * square_length_m  # 미터 단위
                    board_points_3d.append([x, y, 0])
            
            board_points_3d = np.array(board_points_3d, dtype=np.float32)
            
            # 검출된 코너에 해당하는 3D 점들만 선택
            detected_points_3d = []
            detected_points_2d = []
            
            for i, corner_id in enumerate(charuco_ids):
                if corner_id < len(board_points_3d):
                    detected_points_3d.append(board_points_3d[corner_id])
                    detected_points_2d.append(charuco_corners[i])
            
            if len(detected_points_3d) >= 4:
                detected_points_3d = np.array(detected_points_3d, dtype=np.float32)
                detected_points_2d = np.array(detected_points_2d, dtype=np.float32)
                
                # solvePnP로 포즈 추정
                ret, rvec, tvec = cv2.solvePnP(
                    detected_points_3d, detected_points_2d,
                    camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if ret:
                    print(f"\rDEBUG: ChArUco corner-based pose estimation successful", end="")
                    return rvec, tvec, charuco_corners, charuco_ids
        
        # 2. ChArUco 코너 기반 pose 추정 실패 시 ArUco 마커 기반 fallback 전략
        print(f"\rDEBUG: ChArUco corner-based pose estimation failed, trying ArUco marker-based fallback", end="")
        
        # ArUco 마커 검출 (완화된 파라미터 사용)
        try:
            aruco_params = cv2.aruco.DetectorParameters()
            aruco_params.adaptiveThreshWinSizeMin = 3
            aruco_params.adaptiveThreshWinSizeMax = 23
            aruco_params.adaptiveThreshWinSizeStep = 10
            aruco_params.adaptiveThreshConstant = 7
            aruco_params.minMarkerPerimeterRate = 0.003  # 매우 완화된 파라미터
            aruco_params.maxMarkerPerimeterRate = 4.0
            aruco_params.polygonalApproxAccuracyRate = 0.05
            aruco_params.minCornerDistanceRate = 0.003
            aruco_params.minMarkerDistanceRate = 0.003
            aruco_params.minDistanceToBorder = 1
            aruco_params.minOtsuStdDev = 2.0
            aruco_params.perspectiveRemovePixelPerCell = 4
            aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
            aruco_params.maxErroneousBitsInBorderRate = 0.35
            aruco_params.errorCorrectionRate = 0.6
            
            aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, aruco_params)
            corners, ids, rejected = aruco_detector.detectMarkers(gray_enhanced)
        except AttributeError:
            # OpenCV 4.4 이하 버전
            aruco_params = cv2.aruco.DetectorParameters_create()
            aruco_params.adaptiveThreshWinSizeMin = 3
            aruco_params.adaptiveThreshWinSizeMax = 23
            aruco_params.adaptiveThreshWinSizeStep = 10
            aruco_params.adaptiveThreshConstant = 7
            aruco_params.minMarkerPerimeterRate = 0.003
            aruco_params.maxMarkerPerimeterRate = 4.0
            aruco_params.polygonalApproxAccuracyRate = 0.05
            aruco_params.minCornerDistanceRate = 0.003
            aruco_params.minMarkerDistanceRate = 0.003
            aruco_params.minDistanceToBorder = 1
            aruco_params.minOtsuStdDev = 2.0
            aruco_params.perspectiveRemovePixelPerCell = 4
            aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
            aruco_params.maxErroneousBitsInBorderRate = 0.35
            aruco_params.errorCorrectionRate = 0.6
            
            corners, ids, rejected = cv2.aruco.detectMarkers(gray_enhanced, self.aruco_dict, parameters=aruco_params)
        
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
                center_x = marker_col * square_length_m  # 미터 단위
                center_y = marker_row * square_length_m  # 미터 단위
                
                # 마커의 4개 코너를 3D 점으로 생성 (미터 단위)
                half_marker = marker_length_m / 2  # 미터 단위
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
    
    def collect_angles_realtime(self, num_poses: int = 20) -> bool:
        """
        실시간으로 로봇 각도 데이터를 수집합니다.
        
        Args:
            num_poses: 수집할 포즈 개수
            
        Returns:
            bool: 수집 성공 여부
        """
        print(f"\n=== 실시간 각도 수집 모드 ===")
        print(f"수집할 포즈 수: {num_poses}")
        print("로봇을 수동으로 조작하여 다양한 포즈의 각도를 수집합니다.")
        print("각 포즈에서 Enter를 눌러 저장하거나 'q'를 눌러 종료하세요.")
        print("현재 포즈 정보가 실시간으로 표시됩니다.")
        
        collected_poses = 0
        pose_index = 0
        
        while collected_poses < num_poses:
            try:
                # 현재 각도 가져오기
                angles = self.get_robot_angles()
                coords = self.robot.get_coords()
                
                # 실시간 정보 출력
                print(f"\n{'='*50}")
                print(f"=== 실시간 각도 수집 모드 ===")
                print(f"수집할 포즈 수: {num_poses}")
                print(f"현재 수집된 포즈: {collected_poses}")
                print(f"현재 포즈 인덱스: {pose_index}")
                print(f"\n현재 로봇 상태:")
                print(f"  각도: [{angles[0]:.1f}, {angles[1]:.1f}, {angles[2]:.1f}, "
                      f"{angles[3]:.1f}, {angles[4]:.1f}, {angles[5]:.1f}]")
                print(f"  좌표: [{coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}]")
                print(f"\n조작법:")
                print(f"  - Enter: 현재 포즈 저장")
                print(f"  - q: 종료")
                print(f"  - Ctrl+C: 강제 종료")
                print(f"{'='*50}")
                
                # 사용자 입력 대기
                user_input = input("\n명령을 입력하세요: ").strip().lower()
                
                if user_input == 'q':
                    print(f"\n사용자에 의해 중단되었습니다. 수집된 포즈: {collected_poses}")
                    break
                elif user_input == '' or user_input == 's':
                    # 현재 포즈 저장
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    
                    angle_data = {
                        "angles": angles,
                        "coords": coords,
                        "pose_index": pose_index,
                        "timestamp": timestamp
                    }
                    
                    angle_path = self.angles_dir / f"{pose_index:02d}_{timestamp}_angles.json"
                    with open(angle_path, 'w') as f:
                        json.dump(angle_data, f, indent=2)
                    
                    print(f"\n✅ 포즈 {pose_index} 저장 완료: {angle_path}")
                    collected_poses += 1
                    pose_index += 1
                    
                    # 잠시 대기 후 계속
                    input("계속하려면 Enter를 누르세요...")
                
            except KeyboardInterrupt:
                print(f"\n사용자에 의해 중단되었습니다. 수집된 포즈: {collected_poses}")
                break
            except Exception as e:
                print(f"\n오류 발생: {e}")
                input("계속하려면 Enter를 누르세요...")
                continue
        
        print(f"\n=== 실시간 각도 수집 완료 ===")
        print(f"총 {collected_poses}개의 포즈 데이터가 수집되었습니다.")
        return collected_poses >= 10  # 최소 10개 이상 필요
    
    def collect_angles_realtime_simple(self, num_poses: int = 20) -> bool:
        """
        간단한 실시간 각도 수집 모드 (더 안정적인 버전)
        
        Args:
            num_poses: 수집할 포즈 개수
            
        Returns:
            bool: 수집 성공 여부
        """
        print(f"\n=== 실시간 각도 수집 모드 (간단 버전) ===")
        print(f"수집할 포즈 수: {num_poses}")
        print("로봇을 수동으로 조작하여 다양한 포즈의 각도를 수집합니다.")
        print("각 포즈에서 Enter를 눌러 저장하거나 'q'를 눌러 종료하세요.")
        
        collected_poses = 0
        pose_index = 0
        
        while collected_poses < num_poses:
            try:
                # 현재 각도 가져오기
                angles = self.get_robot_angles()
                coords = self.robot.get_coords()
                
                # 현재 상태 출력
                print(f"\n{'='*60}")
                print(f"현재 상태: {collected_poses}/{num_poses} 포즈 수집됨")
                print(f"목표: {num_poses}개 포즈")
                print(f"현재 포즈 인덱스: {pose_index}")
                print(f"\n로봇 상태:")
                print(f"   각도: [{angles[0]:6.1f}, {angles[1]:6.1f}, {angles[2]:6.1f}, "
                      f"{angles[3]:6.1f}, {angles[4]:6.1f}, {angles[5]:6.1f}]")
                print(f"   좌표: [{coords[0]:6.1f}, {coords[1]:6.1f}, {coords[2]:6.1f}]")
                print(f"\n⌨조작법:")
                print(f"   Enter: 현재 포즈 저장")
                print(f"   q: 종료")
                print(f"   Ctrl+C: 강제 종료")
                print(f"{'='*60}")
                
                # 사용자 입력 대기
                user_input = input("\n명령을 입력하세요: ").strip().lower()
                
                if user_input == 'q':
                    print(f"\n🛑 사용자에 의해 중단되었습니다. 수집된 포즈: {collected_poses}")
                    break
                elif user_input == '' or user_input == 's':
                    # 현재 포즈 저장
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    
                    angle_data = {
                        "angles": angles,
                        "coords": coords,
                        "pose_index": pose_index,
                        "timestamp": timestamp
                    }
                    
                    angle_path = self.angles_dir / f"{pose_index:02d}_{timestamp}_angles.json"
                    with open(angle_path, 'w') as f:
                        json.dump(angle_data, f, indent=2)
                    
                    print(f"\n✅ 포즈 {pose_index} 저장 완료!")
                    print(f"   📁 저장 위치: {angle_path}")
                    collected_poses += 1
                    pose_index += 1
                    
                    if collected_poses < num_poses:
                        print(f"\n🎉 {collected_poses}/{num_poses} 포즈 수집 완료!")
                        input("다음 포즈로 계속하려면 Enter를 누르세요...")
                    else:
                        print(f"\n🎉 모든 포즈 수집 완료!")
                
            except KeyboardInterrupt:
                print(f"\n🛑 사용자에 의해 중단되었습니다. 수집된 포즈: {collected_poses}")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                input("계속하려면 Enter를 누르세요...")
                continue
        
        print(f"\n{'='*60}")
        print(f"실시간 각도 수집 완료")
        print(f"총 {collected_poses}개의 포즈 데이터가 수집되었습니다.")
        print(f"{'='*60}")
        
        return collected_poses >= 10  # 최소 10개 이상 필요
    
    def collect_angles_with_camera_feed(self, num_poses: int = 20) -> bool:
        """
        카메라 화면을 보여주면서 실시간 각도 수집 모드 (카메라 캡처 없이)
        
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
        
        # RealSense 카메라 초기화 (1280x800 해상도)
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, 30)  
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        
        pipeline = None
        try:
            pipeline = rs.pipeline()
            profile = pipeline.start(config)
            print("✅ RealSense 카메라 연결 성공 (1280x800)")
        except Exception as e:
            print(f"❌ 카메라 연결 실패: {e}")
            print("카메라 없이 각도 수집을 계속합니다...")
            return self.collect_angles_realtime_simple(num_poses)
        
        # 카메라 내부 파라미터 가져오기 (1280x800 해상도용)
        camera_matrix = np.array([
            [642.947,   0.0,     644.808],
            [0.0,       642.073, 409.050],
            [0.0,       0.0,     1.0]
        ], dtype=np.float64)
        
        # 왜곡 계수 (k1, k2, p1, p2, k3)
        dist_coeffs = np.array([
            -0.05594644322991371,
            0.06878077983856201,
            -0.00011232726683374494,
            0.000743341923225671,
            -0.022005939856171608
        ], dtype=np.float64)
        
        collected_poses = 0
        pose_index = 0
        
        try:
            while collected_poses < num_poses:
                self.robot.release_all_servos()
                # 카메라 프레임 가져오기
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                # 이미지 변환
                color_image = np.asanyarray(color_frame.get_data())
                
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
                    
                    try:
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
                    except AttributeError:
                        # OpenCV 4.4 이하 버전
                        aruco_params = cv2.aruco.DetectorParameters_create()
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
                        
                        corners, ids, rejected = cv2.aruco.detectMarkers(gray_enhanced, self.aruco_dict, parameters=aruco_params)
                    
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
                    cv2.putText(info_image, "Press 's' to save", (10, 250), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.putText(info_image, "No ChArUco detected - cannot save", (10, 250), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # 화면 표시
                cv2.imshow('Cam-to-Gripper Calibration - Real-time', info_image)
                
                # 키보드 입력 확인
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print(f"\n 사용자에 의해 중단되었습니다. 수집된 포즈: {collected_poses}")
                    break
                elif key == ord('s'):
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
                                
                                # ChArUco 데이터 저장
                                charuco_data = {
                                    "rvec": rvec.tolist(),
                                    "tvec": tvec.tolist(),
                                    "pose_index": pose_index,
                                    "timestamp": timestamp,
                                    "charuco_corners_count": len(charuco_corners) if charuco_corners is not None else 0,
                                    "charuco_ids_count": len(charuco_ids) if charuco_ids is not None else 0,
                                    "distance_mm": float(distance * 1000),
                                    "quality_score": float(quality_score),
                                    "detection_method": "charuco_corners" if charuco_corners is not None else "aruco_markers"
                                }
                                charuco_path = self.save_dir / f"{pose_index:02d}_{timestamp}_charuco.json"
                                with open(charuco_path, 'w') as f:
                                    json.dump(charuco_data, f, indent=2)
                                
                                print(f"\n✅ 포즈 {pose_index} 저장 완료!")
                                print(f"  - 거리: {distance*1000:.1f}mm")
                                print(f"  - 품질: {quality_score:.2f}")
                                print(f"  - 검출 방법: {'ChArUco 코너' if charuco_corners is not None else 'ArUco 마커'}")
                                print(f"  - 저장 위치: {angle_path}")
                                
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
    
    def capture_data_at_angles(self, angle_data: Dict[str, Any], pose_index: int) -> bool:
        """
        저장된 각도 데이터를 사용하여 로봇을 이동시키고 이미지를 캡처합니다.
        
        Args:
            angle_data: 각도 데이터 딕셔너리
            pose_index: 포즈 인덱스
            
        Returns:
            bool: 캡처 성공 여부
        """
        try:
            print(f"\n[{pose_index}] 자동 이동 모드")
            
            # 저장된 각도로 로봇 이동
            angles = angle_data["angles"]
            print(f"로봇을 각도 {angles}로 이동 중...")
            
            # 로봇 이동 (속도 50)
            self.robot.send_angles(angles, 50)
            time.sleep(3)  # 이동 완료 대기
            
            # 현재 로봇 포즈 가져오기
            R_base2ee, t_base2ee = self.get_robot_pose()
            
            # 이미지 캡처
            print("이미지 캡처 중...")
            color_path, depth_path, intrinsics_path = capture_d455_images(
                save_dir=str(self.save_dir),
                rgb_size=(848, 480),
                depth_size=(848, 480)
            )
            
            # 카메라 내부 파라미터 로드
            camera_matrix, dist_coeffs = self._load_camera_intrinsics(intrinsics_path)
            
            # 이미지에서 ChArUco 검출
            image = cv2.imread(color_path)
            charuco_pose = self.detect_charuco_pose(image, camera_matrix, dist_coeffs)
            
            if charuco_pose is None:
                print(f"[{pose_index}] ChArUco 검출 실패")
                print("ChArUco 보드가 카메라에 잘 보이는지 확인하고 다시 시도해주세요.")
                return False
            
            rvec, tvec, charuco_corners, charuco_ids = charuco_pose
            
            # 거리 정보 출력 (디버깅용)
            distance = np.linalg.norm(tvec)
            print(f"ChArUco 보드까지의 거리: {distance*1000:.1f}mm")
            
            # 데이터 저장
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
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
            
            # ChArUco 포즈 저장 (카메라 -> 타겟)
            charuco_data = {
                "rvec": rvec.tolist(),
                "tvec": tvec.tolist(),
                "pose_index": pose_index,
                "timestamp": timestamp
            }
            charuco_path = self.save_dir / f"{pose_index:02d}_{timestamp}_charuco.json"
            
            with open(charuco_path, 'w') as f:
                json.dump(charuco_data, f, indent=2)
            
            print(f"[{pose_index}] ✅ 저장 완료")
            print(f"  - 이미지: {color_path}")
            print(f"  - 깊이: {depth_path}")
            print(f"  - 로봇 포즈: {pose_path}")
            print(f"  - ChArUco 포즈: {charuco_path}")
            
            return True
            
        except Exception as e:
            print(f"[{pose_index}] 오류 발생: {e}")
            return False
    
    def perform_automatic_calibration(self) -> bool:
        """
        저장된 각도 데이터를 사용하여 자동으로 캘리브레이션을 수행합니다.
        
        Returns:
            bool: 캘리브레이션 성공 여부
        """
        print(f"\n=== 자동 캘리브레이션 모드 ===")
        
        # 저장된 각도 파일들 로드
        angle_files = list(self.angles_dir.glob("*_angles.json"))
        
        if len(angle_files) == 0:
            print("저장된 각도 데이터가 없습니다. 먼저 실시간 각도 수집을 수행해주세요.")
            return False
        
        # 파일 정렬 (인덱스 순서대로)
        angle_files.sort(key=lambda x: int(x.stem.split('_')[0]))
        
        print(f"총 {len(angle_files)}개의 각도 데이터를 사용하여 자동 캘리브레이션을 시작합니다...")
        
        successful_captures = 0
        
        for i, angle_file in enumerate(angle_files):
            print(f"\n[{i+1}/{len(angle_files)}] 각도 데이터 처리 중...")
            
            # 각도 데이터 로드
            with open(angle_file, 'r') as f:
                angle_data = json.load(f)
            
            pose_index = angle_data["pose_index"]
            
            if self.capture_data_at_angles(angle_data, pose_index):
                successful_captures += 1
                print(f"✅ 포즈 {pose_index} 캡처 성공 ({successful_captures}/{len(angle_files)})")
            else:
                print(f"❌ 포즈 {pose_index} 캡처 실패")
                
                # 재시도 옵션
                retry = input("이 포즈를 다시 시도하시겠습니까? (y/n): ").lower().strip()
                if retry == 'y':
                    i -= 1  # 같은 파일을 다시 처리
                    continue
        
        print(f"\n=== 자동 캡처 완료 ===")
        print(f"성공: {successful_captures}/{len(angle_files)}")
        
        if successful_captures >= 10:  # 최소 10개 이상의 포즈가 필요
            return self.calculate_transformation_matrix()
        else:
            print("충분한 데이터가 수집되지 않았습니다.")
            return False
    
    def capture_data_at_manual_pose(self, pose_index: int) -> bool:
        """
        수동으로 조작된 로봇 포즈에서 이미지와 포즈 데이터를 캡처합니다.
        
        Args:
            pose_index: 포즈 인덱스
            
        Returns:
            bool: 캡처 성공 여부
        """
        try:
            print(f"\n[{pose_index}] 수동 조작 모드")
            
            # 로봇을 수동 조작 모드로 전환
            self.robot.release_all_servos()
            
            # 사용자 입력 대기
            input("로봇을 원하는 위치로 이동한 후 Enter를 눌러주세요...")
            
            # 로봇을 고정 모드로 전환
            self.robot.power_on()
            time.sleep(2)  # 로봇 안정화 대기
            
            # 현재 로봇 포즈 가져오기
            R_base2ee, t_base2ee = self.get_robot_pose()
            
            # 이미지 캡처
            print("이미지 캡처 중...")
            color_path, depth_path, intrinsics_path = capture_d455_images(
                save_dir=str(self.save_dir),
                rgb_size=(848, 480),
                depth_size=(848, 480)
            )
            
            # 카메라 내부 파라미터 로드
            camera_matrix, dist_coeffs = self._load_camera_intrinsics(intrinsics_path)
            
            # 이미지에서 ChArUco 검출
            image = cv2.imread(color_path)
            charuco_pose = self.detect_charuco_pose(image, camera_matrix, dist_coeffs)
            
            if charuco_pose is None:
                print(f"[{pose_index}] ChArUco 검출 실패")
                print("ChArUco 보드가 카메라에 잘 보이는지 확인하고 다시 시도해주세요.")
                return False
            
            rvec, tvec, charuco_corners, charuco_ids = charuco_pose
            
            # 거리 정보 출력 (디버깅용)
            distance = np.linalg.norm(tvec)
            print(f"ChArUco 보드까지의 거리: {distance*1000:.1f}mm")
            
            # 데이터 저장
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            # 로봇 포즈 저장
            pose_data = {
                "R_base2ee": R_base2ee.tolist(),
                "t_base2ee": t_base2ee.tolist(),
                "pose_index": pose_index,
                "timestamp": timestamp
            }
            pose_path = self.poses_dir / f"{pose_index:02d}_{timestamp}_pose.json"
            
            with open(pose_path, 'w') as f:
                json.dump(pose_data, f, indent=2)
            
            # ChArUco 포즈 저장 (카메라 -> 타겟)
            charuco_data = {
                "rvec": rvec.tolist(),
                "tvec": tvec.tolist(),
                "pose_index": pose_index,
                "timestamp": timestamp
            }
            charuco_path = self.save_dir / f"{pose_index:02d}_{timestamp}_charuco.json"
            
            with open(charuco_path, 'w') as f:
                json.dump(charuco_data, f, indent=2)
            
            print(f"[{pose_index}] ✅ 저장 완료")
            print(f"  - 이미지: {color_path}")
            print(f"  - 깊이: {depth_path}")
            print(f"  - 로봇 포즈: {pose_path}")
            print(f"  - ChArUco 포즈: {charuco_path}")
            
            return True
            
        except Exception as e:
            print(f"[{pose_index}] 오류 발생: {e}")
            return False
    
    def perform_manual_calibration(self, num_poses: int = 20) -> bool:
        """
        수동 조작을 통한 Cam-to-Gripper calibration을 수행합니다.
        
        Args:
            num_poses: 캡처할 포즈 개수
            
        Returns:
            bool: 캘리브레이션 성공 여부
        """
        print(f"\n=== 수동 Eye-in-Hand Calibration 시작 ===")
        print(f"캡처할 포즈 수: {num_poses}")
        print("각 포즈에서 로봇을 수동으로 조작하여 ChArUco 보드를 다양한 각도에서 촬영합니다.")
        
        successful_captures = 0
        current_pose_index = 0
        
        while successful_captures < num_poses:
            print(f"\n[{successful_captures}/{num_poses}] 포즈 캡처 중...")
            
            if self.capture_data_at_manual_pose(current_pose_index):
                successful_captures += 1
                print(f"✅ 포즈 {successful_captures} 캡처 성공 ({successful_captures}/{num_poses})")
                # 검출 성공 시 자동으로 다음 포즈로 진행
                current_pose_index += 1
            else:
                print(f"❌ 포즈 {current_pose_index} 캡처 실패")
                
                # 재시도 옵션 (검출 실패 시에만 질문)
                retry = input("이 포즈를 다시 시도하시겠습니까? (y/n/q): ").lower().strip()
                if retry == 'q':
                    print("캘리브레이션을 중단합니다.")
                    break
                elif retry == 'y':
                    continue  # 같은 인덱스로 다시 시도
                else:
                    print("이 포즈를 건너뛰고 다음으로 진행합니다.")
                    current_pose_index += 1
        
        print(f"\n=== 수동 캡처 완료 ===")
        print(f"성공: {successful_captures}/{num_poses}")
        
        if successful_captures >= 10:  # 최소 10개 이상의 포즈가 필요
            return self.calculate_transformation_matrix()
        else:
            print("충분한 데이터가 수집되지 않았습니다.")
            return False
    
    def calculate_transformation_matrix(self) -> bool:
        """
        수집된 데이터를 사용하여 Eye-in-Hand 변환 행렬을 계산합니다.
        """
        print("\n=== Eye-in-Hand 변환 행렬 계산 중 ===")
        
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
                # ChArUco 포즈 로드 (카메라 -> 타겟)
                with open(charuco_file, 'r') as f:
                    charuco_data = json.load(f)
                
                rvec = np.array(charuco_data["rvec"])
                tvec = np.array(charuco_data["tvec"])
                
                # 회전 벡터를 회전 행렬로 변환
                camera_rotation, _ = cv2.Rodrigues(rvec)
                
                # 카메라 -> ChArUco 타겟 (미터 단위)
                T_cam2target = np.eye(4)
                T_cam2target[:3, :3] = camera_rotation
                T_cam2target[:3, 3] = tvec.flatten()
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
                print(f"포즈 {i+1}: {detection_method}, 거리={distance_mm:.1f}mm")
            
            # Hand-Eye calibration 수행
            print("\nHand-Eye calibration 계산 중...")
            R_ee2cam, t_ee2cam = self._solve_hand_eye_calibration(
                T_cam2target_list, T_base2ee_list
            )
            
            if R_ee2cam is not None:
                # 결과는 미터 단위로 반환됨
                t_ee2cam_norm = np.linalg.norm(t_ee2cam)
                print(f"t_ee2cam 크기 (m): {t_ee2cam_norm:.3f}")
                
                # 미터 단위로 저장 (표준 단위)
                print(f"단위: 미터 (표준 단위)")
                
                # 변환 행렬 생성
                T_ee2cam = np.eye(4)
                T_ee2cam[:3, :3] = R_ee2cam
                T_ee2cam[:3, 3] = t_ee2cam.flatten()
                
                # 결과 저장 (미터 단위)
                calibration_result = {
                    "R_ee2cam": R_ee2cam.tolist(),
                    "t_ee2cam": t_ee2cam.flatten().tolist(),  # 미터 단위로 저장
                    "num_poses": len(T_base2ee_list),
                    "timestamp": datetime.now().isoformat(),
                    "description": "End-Effector -> 카메라 변환 행렬 (Eye-in-Hand calibration, 미터 단위)",
                    "coordinate_transformation": {
                        "applied": True,
                        "description": "Z_cam → X_robot, -X_cam → Y_robot, -Y_cam → Z_robot",
                        "transformation_matrix": [
                            [0, 0, 1],
                            [-1, 0, 0],
                            [0, -1, 0]
                        ]
                    }
                }
                
                result_path = self.save_dir / "ee2cam.json"
                with open(result_path, 'w') as f:
                    json.dump(calibration_result, f, indent=2)
                
                # numpy 배열을 안전하게 float로 변환
                t_x = float(t_ee2cam.flatten()[0])
                t_y = float(t_ee2cam.flatten()[1])
                t_z = float(t_ee2cam.flatten()[2])
                
                print(f"✅ Eye-in-Hand 캘리브레이션 완료!")
                print(f"결과 저장: {result_path}")
                print(f"End-Effector -> 카메라 변환 행렬:")
                print(f"R:\n{R_ee2cam}")
                print(f"t (m): [{t_x:.3f}, {t_y:.3f}, {t_z:.3f}]")
                
                return True
            else:
                print("❌ 변환 행렬 계산 실패")
                return False
                
        except Exception as e:
            print(f"캘리브레이션 계산 중 오류: {e}")
            return False
    

    
    def _solve_hand_eye_calibration(self, 
                                   T_cam2target_list: List[np.ndarray],
                                   T_base2ee_list: List[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Hand-Eye calibration 방정식을 해결합니다.
        
        Args:
            T_cam2target_list: 카메라 포즈 리스트 (카메라 -> ChArUco 타겟) - 미터 단위
            T_base2ee_list: 로봇 포즈 리스트 (베이스 -> End-Effector) - 미터 단위
            
        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: (R_ee2cam, t_ee2cam) - 미터 단위
        """
        try:
            # OpenCV의 calibrateHandEye 함수 사용
            camera_rotations = [pose[:3, :3] for pose in T_cam2target_list]
            camera_translations = [pose[:3, 3] for pose in T_cam2target_list]
            ee_rotations = [pose[:3, :3] for pose in T_base2ee_list]
            ee_translations = [pose[:3, 3] for pose in T_base2ee_list]
            
            # 충분한 데이터가 있는지 확인
            if len(T_cam2target_list) < 3:
                print("충분한 포즈 데이터가 없습니다.")
                return None, None
            
            # Hand-Eye calibration 수행 (모든 입력은 미터 단위)
            R_ee2cam, t_ee2cam = cv2.calibrateHandEye(
                ee_rotations, ee_translations,
                camera_rotations, camera_translations,
                method=cv2.CALIB_HAND_EYE_TSAI
            )
            
            # 좌표계 차이를 반영한 정렬 행렬 (카메라 좌표 → 로봇 gripper 좌표)
            T_camera_to_robot_coord = np.array([
                [ 0,  0,  1],
                [-1,  0,  0],
                [ 0, -1,  0]
            ])  # Z_cam → X_robot, -X_cam → Y_robot, -Y_cam → Z_robot

            # 보정 적용: R_ee2cam 보정
            R_corrected = T_camera_to_robot_coord @ R_ee2cam
            t_corrected = T_camera_to_robot_coord @ t_ee2cam
            
            print(f"원본 t_ee2cam: {t_ee2cam.flatten()}")
            print(f"보정된 t_ee2cam: {t_corrected.flatten()}")
            
            return R_corrected, t_corrected  # 보정된 결과 반환
            
        except Exception as e:
            print(f"Hand-Eye calibration 계산 오류: {e}")
            return None, None
    
    def test_calibration(self) -> bool:
        """
        캘리브레이션 결과를 테스트합니다.
        """
        print("\n=== Cam-to-Gripper 캘리브레이션 테스트 ===")
        
        try:
            # 캘리브레이션 결과 로드
            result_path = self.save_dir / "ee2cam.json"
            if not result_path.exists():
                print("캘리브레이션 결과 파일이 없습니다.")
                return False
            
            with open(result_path, 'r') as f:
                calibration_data = json.load(f)
            
            R_ee2cam = np.array(calibration_data["R_ee2cam"])
            t_ee2cam = np.array(calibration_data["t_ee2cam"])  # 미터 단위
            
            print("캘리브레이션 결과:")
            print(f"R_ee2cam:\n{R_ee2cam}")
            
            # 좌표축 변환 정보 확인
            if "coordinate_transformation" in calibration_data:
                coord_info = calibration_data["coordinate_transformation"]
                print(f"\n좌표축 변환 적용됨:")
                print(f"설명: {coord_info['description']}")
                print(f"변환 행렬:\n{np.array(coord_info['transformation_matrix'])}")
            
            # t_ee2cam 단위 확인
            t_ee2cam_norm = np.linalg.norm(t_ee2cam)
            print(f"\nt_ee2cam 크기 (m): {t_ee2cam_norm:.3f}")
            
            # numpy 배열을 안전하게 float로 변환
            t_x = float(t_ee2cam.flatten()[0])
            t_y = float(t_ee2cam.flatten()[1])
            t_z = float(t_ee2cam.flatten()[2])
            print(f"t_ee2cam (m): [{t_x:.3f}, {t_y:.3f}, {t_z:.3f}]")
            
            # 테스트 이미지 캡처
            print("\n테스트 이미지 캡처 중...")
            color_path, depth_path, intrinsics_path = capture_d455_images(
                save_dir=str(self.save_dir),
                rgb_size=(848, 480),
                depth_size=(848, 480)
            )
            
            # 카메라 내부 파라미터 로드
            camera_matrix, dist_coeffs = self._load_camera_intrinsics(intrinsics_path)
            
            # ChArUco 검출
            image = cv2.imread(color_path)
            charuco_pose = self.detect_charuco_pose(image, camera_matrix, dist_coeffs)
            
            if charuco_pose is None:
                print("테스트에서 ChArUco 검출 실패")
                return False
            
            rvec, tvec, charuco_corners, charuco_ids = charuco_pose
            
            # 카메라 -> ChArUco 변환 (미터 단위)
            R_cam2charuco, _ = cv2.Rodrigues(rvec)
            t_cam2charuco = tvec  # 미터 단위
            
            # 현재 로봇 포즈 (베이스 -> Gripper)
            robot_coords = self.robot.get_coords()
            robot_position_mm = np.array(robot_coords[0:3])
            robot_rotation_deg = np.array(robot_coords[3:6])
            
            # numpy 배열을 안전하게 float로 변환
            robot_x = float(robot_position_mm.flatten()[0])
            robot_y = float(robot_position_mm.flatten()[1])
            robot_z = float(robot_position_mm.flatten()[2])
            robot_rx = float(robot_rotation_deg.flatten()[0])
            robot_ry = float(robot_rotation_deg.flatten()[1])
            robot_rz = float(robot_rotation_deg.flatten()[2])
            
            print(f"\n실제 로봇 위치 (베이스 기준):")
            print(f"End-Effector 위치: {robot_x:.1f}, {robot_y:.1f}, {robot_z:.1f}mm")
            print(f"End-Effector 회전: {robot_rx:.1f}, {robot_ry:.1f}, {robot_rz:.1f}도")
            
            # 캘리브레이션을 통한 계산
            # 카메라 -> End-Effector 변환 (이미 미터 단위)
            # 좌표축 변환을 고려한 계산
            charuco_in_ee = R_ee2cam @ t_cam2charuco + t_ee2cam.reshape(3, 1)
            charuco_in_ee_mm = charuco_in_ee.flatten() * 1000  # m -> mm (표시용)
            
            # numpy 배열을 안전하게 float로 변환
            charuco_x = float(charuco_in_ee_mm.flatten()[0])
            charuco_y = float(charuco_in_ee_mm.flatten()[1])
            charuco_z = float(charuco_in_ee_mm.flatten()[2])
            
            print(f"\n캘리브레이션을 통한 ChArUco 위치 계산:")
            print(f"카메라에서 계산된 ChArUco 위치 (End-Effector 기준): {charuco_x:.1f}, {charuco_y:.1f}, {charuco_z:.1f}mm")
            
            # 거리 계산
            distance = np.linalg.norm(charuco_in_ee_mm)
            distance_float = float(distance)
            print(f"카메라에서 계산된 거리: {distance_float:.1f}mm")
            
            # 테스트 결과 출력
            print(f"\n캘리브레이션 테스트 결과:")
            print(f"End-Effector -> 카메라 변환 행렬이 성공적으로 계산되었습니다.")
            print(f"이제 카메라에서 감지된 객체의 픽셀 좌표를 로봇 End-Effector 기준 3D 좌표로 변환할 수 있습니다.")
            
            return True
            
        except Exception as e:
            print(f"캘리브레이션 테스트 오류: {e}")
            return False
    



def main():
    """
    메인 함수
    """
    print("Cam-to-Gripper Calibration for MyCobot280")
    print("=" * 50)
    
    # 캘리브레이터 초기화 (실제 보드 크기에 맞춤)
    calibrator = CamToGripperCalibrator(
        robot_port="/dev/ttyACM0",
        robot_baud=115200,
        save_dir="/home/ros/llm_robot/data/Calibration/Eye-in-Hand9",
        charuco_squares_x=9, 
        charuco_squares_y=6,   
        charuco_square_length=28.0, #/1000,  # mm (실제 측정값)
        charuco_marker_length=21.0, #/1000,  # mm (실제 측정값)
        min_charuco_corners=15,  # 최소 ChArUco 코너 개수 (더 완화)
        min_detection_confidence=0.2,  # 최소 검출 신뢰도 (더욱 완화)
        max_distance_threshold=0.7)  # 최대 거리 임계값 (미터)
    
    
    try:
        print("\n=== 캘리브레이션 옵션 ===")
        print("1. 실시간 각도 수집 모드 (카메라 화면 포함)")
        print("2. 자동 캘리브레이션 모드 (저장된 각도 사용)")
        print("3. 수동 캘리브레이션 모드 (기존 방식)")
        print("4. 테스트만 실행")
        print("5. 종료")
        
        choice = input("선택 (1/2/3/4/5): ").strip()
        
        if choice == "1":
            # 실시간 각도 수집 모드 (카메라 화면 포함)
            num_poses = int(input("\n수집할 포즈 수를 입력하세요 (권장: 25-30): ") or "25")
            print(f"\n{num_poses}개의 포즈로 실시간 각도 수집을 시작합니다...")
            print("카메라 화면이 표시되며, ChArUco 검출 상태를 실시간으로 확인할 수 있습니다.")
            print("각 포즈에서 's'를 눌러 저장하거나 'q'를 눌러 종료하세요.")
            
            collection_success = calibrator.collect_angles_with_camera_feed(num_poses)
            
            if collection_success:
                print("\n🎉 실시간 각도 수집 완료!")
                print("\n📊 수집된 데이터로 ee2cam.json을 자동 계산합니다...")
                calibration_success = calibrator.calculate_transformation_matrix()
                if calibration_success:
                    print("\n🎉 Cam-to-Gripper 캘리브레이션 완료!")
                    test_choice = input("\n캘리브레이션 테스트를 실행하시겠습니까? (y/n): ").lower().strip()
                    if test_choice == 'y':
                        calibrator.test_calibration()
                else:
                    print("\n❌ 캘리브레이션 실패")
            else:
                print("\n❌ 각도 수집 실패")
                
        elif choice == "2":
            # 자동 캘리브레이션 모드
            print("\n저장된 각도 데이터를 사용하여 자동 캘리브레이션을 시작합니다...")
            calibration_success = calibrator.perform_automatic_calibration()
            
            if calibration_success:
                print("\n�� Cam-to-Gripper 캘리브레이션 완료!")
                test_choice = input("\n캘리브레이션 테스트를 실행하시겠습니까? (y/n): ").lower().strip()
                if test_choice == 'y':
                    calibrator.test_calibration()
            else:
                print("\n❌ 캘리브레이션 실패")
                
        elif choice == "3":
            # 수동 캘리브레이션 모드 (기존 방식)
            num_poses = int(input("\n캡처할 포즈 수를 입력하세요 (권장: 25-30): ") or "25")
            print(f"\n{num_poses}개의 포즈로 수동 Cam-to-Gripper 캘리브레이션을 시작합니다...")
            calibration_success = calibrator.perform_manual_calibration(num_poses)
            
            if calibration_success:
                print("\n🎉 Cam-to-Gripper 캘리브레이션 완료!")
                test_choice = input("\n캘리브레이션 테스트를 실행하시겠습니까? (y/n): ").lower().strip()
                if test_choice == 'y':
                    calibrator.test_calibration()
            else:
                print("\n❌ 캘리브레이션 실패")
                
        elif choice == "4":
            # 테스트만 실행
            calibrator.test_calibration()
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