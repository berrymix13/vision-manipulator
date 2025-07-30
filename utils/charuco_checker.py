#!/usr/bin/env python3
"""
ChArUco 검출 실시간 테스트 스크립트

RealSense D455 카메라에서 실시간으로 ChArUco 마커를 검출하고 결과를 시각화합니다.
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import time
from pathlib import Path
from typing import List, Tuple, Optional


class ChArUcoDetector:
    """
    실시간 ChArUco 검출 클래스
    """
    
    def __init__(self, 
                 squares_x: int = 8,
                 squares_y: int = 11,
                 square_length: float = 23.0,  # mm
                 marker_length: float = 17.0,  # mm
                 camera_intrinsics_path: str = "/home/ros/llm_robot/data/Calibration/Eye-in-Hand6/intrinsics/2025-07-28_20-56-40.json"):
        """
        ChArUco 검출기 초기화
        
        Args:
            squares_x: 가로 방향 사각형 개수
            squares_y: 세로 방향 사각형 개수
            square_length: 사각형 한 변의 길이 (mm)
            marker_length: ArUco 마커 한 변의 길이 (mm)
            camera_intrinsics_path: 카메라 내부 파라미터 파일 경로
        """
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length = square_length / 1000.0  # mm를 m로 변환
        self.marker_length = marker_length / 1000.0  # mm를 m로 변환
        
        # 카메라 내부 파라미터 로드
        self.camera_matrix, self.dist_coeffs = self._load_camera_intrinsics(camera_intrinsics_path)
        
        # ChArUco 딕셔너리 생성
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.charuco_board = cv2.aruco.CharucoBoard(
            size=(squares_x, squares_y),
            squareLength=self.square_length,
            markerLength=self.marker_length,
            dictionary=self.aruco_dict
        )
        
        # RealSense 파이프라인 설정
        self.pipe = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        
        # 검출 통계
        self.detection_count = 0
        self.total_frames = 0
        
        print(f"ChArUco 검출기 초기화 완료")
        print(f"ChArUco 크기: {squares_x}x{squares_y}")
        print(f"사각형 길이: {square_length:.1f}mm")
        print(f"마커 길이: {marker_length:.1f}mm")
    
    def _load_camera_intrinsics(self, camera_intrinsics_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        카메라 내부 파라미터를 로드합니다.
        """
        try:
            fs = cv2.FileStorage(camera_intrinsics_path, cv2.FILE_STORAGE_READ)
            camera_matrix = fs.getNode("camera_matrix").mat()
            dist_coeffs = fs.getNode("dist_coeffs").mat()
            fs.release()
            print(f"카메라 내부 파라미터 로드 완료")
            return camera_matrix, dist_coeffs
        except Exception as e:
            print(f"카메라 내부 파라미터 로드 실패: {e}")
            # 기본 카메라 행렬 사용
            camera_matrix = np.array([
                [848, 0, 424],
                [0, 848, 240],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((5, 1), dtype=np.float64)
            return camera_matrix, dist_coeffs
    
    def detect_charuco(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        이미지에서 ChArUco를 검출합니다.
        
        Returns:
            Optional[Tuple]: (charuco_corners, charuco_ids, pose) 또는 None
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        try:
            charuco_params = cv2.aruco.CharucoParameters()
            # 검출 민감도 향상을 위한 파라미터 조정
            charuco_params.minMarkers = 1  # 최소 마커 수를 더 낮춤
            charuco_params.adaptiveThreshWinSizeMin = 3
            charuco_params.adaptiveThreshWinSizeMax = 23
            charuco_params.adaptiveThreshWinSizeStep = 10
            charuco_params.adaptiveThreshConstant = 7
            charuco_params.minCornerDistanceRate = 0.03
            charuco_params.minMarkerDistanceRate = 0.03
            charuco_params.minDistanceToBorder = 3
            
            charuco_detector = cv2.aruco.CharucoDetector(self.charuco_board, charuco_params)
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
            
            # 디버그 정보 출력
            print(f"검출 결과 - 코너: {len(charuco_corners) if charuco_corners is not None else 'None'}, ID: {len(charuco_ids) if charuco_ids is not None else 'None'}")
            if marker_corners is not None:
                print(f"ArUco 마커 검출: {len(marker_corners)}개")
            
            # None 체크 및 유효성 검사
            if (charuco_corners is not None and charuco_ids is not None and
                len(charuco_corners) > 4 and len(charuco_corners) == len(charuco_ids)):
                
                # ChArUco 포즈 계산
                ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, self.charuco_board,
                    self.camera_matrix, self.dist_coeffs
                )
                
                if ret:
                    print(f"검출된 코너 수: {len(charuco_corners)}")
                    return charuco_corners, charuco_ids, (rvec, tvec)
                else:
                    print("포즈 계산 실패")
            else:
                if charuco_corners is not None:
                    print(f"유효하지 않은 검출 결과 - 코너: {len(charuco_corners)}, ID: {len(charuco_ids) if charuco_ids is not None else 'None'}")
                    
                    # 코너와 ID 개수가 일치하지 않을 때 처리
                    if charuco_corners is not None and charuco_ids is not None:
                        min_len = min(len(charuco_corners), len(charuco_ids))
                        if min_len > 4:
                            # 개수가 일치하는 부분만 사용
                            charuco_corners = charuco_corners[:min_len]
                            charuco_ids = charuco_ids[:min_len]
                            
                            # ChArUco 포즈 계산
                            ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                                charuco_corners, charuco_ids, self.charuco_board,
                                self.camera_matrix, self.dist_coeffs
                            )
                            
                            if ret:
                                print(f"조정된 코너 수: {len(charuco_corners)}")
                                return charuco_corners, charuco_ids, (rvec, tvec)
        except AttributeError:
            # 이전 OpenCV 버전용 fallback
            try:
                # ArUco 마커 검출
                aruco_params = cv2.aruco.DetectorParameters()
                aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, aruco_params)
                corners, ids, rejected = aruco_detector.detectMarkers(gray)
            except AttributeError:
                # OpenCV 4.4 이하 버전
                aruco_params = cv2.aruco.DetectorParameters_create()
                corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=aruco_params)
            
            if len(corners) > 0:
                # ChArUco 코너 검출 (deprecated 함수 사용)
                try:
                    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                        corners, ids, gray, self.charuco_board
                    )
                    
                    if ret and len(charuco_corners) > 4 and len(charuco_corners) == len(charuco_ids):
                        # ChArUco 포즈 계산
                        ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                            charuco_corners, charuco_ids, self.charuco_board,
                            self.camera_matrix, self.dist_coeffs
                        )
                        
                        if ret:
                            print(f"검출된 코너 수: {len(charuco_corners)}")
                            return charuco_corners, charuco_ids, (rvec, tvec)
                except AttributeError:
                    print("Warning: ChArUco 함수를 사용할 수 없습니다.")
                    # 대안: ArUco 마커만 사용한 포즈 추정
                    if len(corners) >= 4:
                        # ArUco 마커의 3D 포인트 생성
                        marker_points = np.array([
                            [-self.marker_length/2, -self.marker_length/2, 0],
                            [self.marker_length/2, -self.marker_length/2, 0],
                            [self.marker_length/2, self.marker_length/2, 0],
                            [-self.marker_length/2, self.marker_length/2, 0]
                        ], dtype=np.float32)
                        
                        # 첫 번째 마커의 포즈 추정
                        ret, rvec, tvec = cv2.solvePnP(
                            marker_points, corners[0], self.camera_matrix, self.dist_coeffs
                        )
                        if ret:
                            return corners[0], ids[0], (rvec, tvec)
        except Exception as e:
            print(f"ChArUco 검출 오류: {e}")
            return None
        
        return None
    
    def test_different_sizes(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        다양한 ChArUco 크기로 테스트합니다.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found_sizes = []
        
        # 테스트할 크기들
        test_sizes = [
            (5, 6), (5, 7), (6, 5), (6, 6), (6, 7), (6, 8),
            (7, 5), (7, 6), (7, 7), (7, 8), (8, 6), (8, 7)
        ]
        
        for size in test_sizes:
            test_board = cv2.aruco.CharucoBoard(
                size=size,
                squareLength=self.square_length,
                markerLength=self.marker_length,
                dictionary=self.aruco_dict
            )
            
            try:
                charuco_params = cv2.aruco.CharucoParameters()
                charuco_detector = cv2.aruco.CharucoDetector(test_board, charuco_params)
                charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
                
                # None 체크 추가
                if charuco_corners is not None and len(charuco_corners) > 4:
                    found_sizes.append(size)
            except AttributeError:
                # 이전 OpenCV 버전용 fallback
                try:
                    # ArUco 마커 검출
                    aruco_params = cv2.aruco.DetectorParameters()
                    aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, aruco_params)
                    corners, ids, rejected = aruco_detector.detectMarkers(gray)
                except AttributeError:
                    # OpenCV 4.4 이하 버전
                    aruco_params = cv2.aruco.DetectorParameters_create()
                    corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=aruco_params)
                
                if len(corners) > 0:
                    # ChArUco 코너 검출 (deprecated 함수 사용)
                    try:
                        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                            corners, ids, gray, test_board
                        )
                        
                        if ret and len(charuco_corners) > 4:
                            found_sizes.append(size)
                    except AttributeError:
                        # OpenCV 4.5 미만에서는 ArUco 마커만으로 크기 확인
                        if len(corners) >= 4:
                            found_sizes.append(size)
        
        return found_sizes
    
    def draw_detection_info(self,
                        image: np.ndarray,
                        charuco_corners: Optional[np.ndarray],
                        charuco_ids: Optional[np.ndarray],
                        pose: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        검출 결과를 이미지에 그립니다.
        """
        rvec, tvec = pose
        
        # ChArUco 코너 그리기 (안전한 처리)
        try:
            # 유효성 검사: corners와 ids의 개수가 정확히 일치하는지 확인
            if (charuco_corners is not None and charuco_ids is not None and 
                len(charuco_corners) == len(charuco_ids) and len(charuco_corners) > 0):
                cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)
            else:
                # 개수가 일치하지 않으면 그리지 않음
                print(f"Warning: 코너({len(charuco_corners) if charuco_corners is not None else 'None'}) != ID({len(charuco_ids) if charuco_ids is not None else 'None'}) - 그리기 건너뜀")
        except cv2.error as e:
            # OpenCV 내부 assertion 에러 처리
            print(f"Warning: drawDetectedCornersCharuco 실패 - {e}")
        except Exception as e:
            print(f"마커 그리기 오류: {e}")
        
        # 좌표축 그리기
        axis_length = 0.1  # 10cm
        axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
        axis_points, _ = cv2.projectPoints(axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        
        # X축 (빨간색)
        cv2.line(image, tuple(axis_points[0].ravel().astype(int)), 
                tuple(axis_points[1].ravel().astype(int)), (0, 0, 255), 3)
        # Y축 (초록색)
        cv2.line(image, tuple(axis_points[0].ravel().astype(int)), 
                tuple(axis_points[2].ravel().astype(int)), (0, 255, 0), 3)
        # Z축 (파란색)
        cv2.line(image, tuple(axis_points[0].ravel().astype(int)), 
                tuple(axis_points[3].ravel().astype(int)), (255, 0, 0), 3)
        
        # 거리 정보 표시
        distance = np.linalg.norm(tvec)
        cv2.putText(image, f"Distance: {distance*1000:.1f}mm", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 검출된 코너 수 표시
        if charuco_corners is not None:
            cv2.putText(image, f"Corners: {len(charuco_corners)}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    def generate_charuco_board_image(self, output_path: str = "charuco_board.png", 
                                   image_size: Tuple[int, int] = (800, 600)) -> None:
        """
        ChArUco 보드 이미지를 생성합니다.
        """
        board_image = self.charuco_board.generateImage(image_size)
        cv2.imwrite(output_path, board_image)
        print(f"ChArUco 보드 이미지 생성됨: {output_path}")
    
    def run_realtime_detection(self):
        """
        실시간 ChArUco 검출을 실행합니다.
        """
        try:
            # RealSense 파이프라인 시작
            profile = self.pipe.start(self.config)
            print("RealSense 카메라 시작됨")
            
            # 카메라 안정화 대기
            time.sleep(3)
            
            # Depth 정렬
            align = rs.align(rs.stream.color)
            
            print("\n=== 실시간 ChArUco 검출 시작 ===")
            print("ESC: 종료, S: 다른 크기 테스트, R: 통계 리셋, C: 이미지 저장, G: 보드 이미지 생성")
            
            while True:
                # 프레임 대기
                frames = self.pipe.wait_for_frames(timeout_ms=5000)
                aligned_frames = align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                if not color_frame:
                    continue
                
                # 이미지 변환
                color_image = np.asanyarray(color_frame.get_data())
                self.total_frames += 1
                
                # ChArUco 검출
                result = self.detect_charuco(color_image)
                
                if result is not None:
                    charuco_corners, charuco_ids, pose = result
                    self.detection_count += 1
                    
                    # 검출 결과 그리기
                    color_image = self.draw_detection_info(color_image, charuco_corners, charuco_ids, pose)
                    
                    # 검출 성공 표시
                    cv2.putText(color_image, "CHARUCO DETECTED!", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # 검출 실패 표시
                    cv2.putText(color_image, "NO CHARUCO", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 통계 정보 표시
                detection_rate = (self.detection_count / self.total_frames * 100) if self.total_frames > 0 else 0
                cv2.putText(color_image, f"Detection Rate: {detection_rate:.1f}%", (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(color_image, f"Frames: {self.total_frames}", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(color_image, f"Detections: {self.detection_count}", (10, 170), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # ChArUco 크기 정보 표시
                cv2.putText(color_image, f"Size: {self.squares_x}x{self.squares_y}", (10, 190), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 이미지 표시
                cv2.imshow('ChArUco Detection Test', color_image)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('s'):  # S키: 다른 크기 테스트
                    self.test_other_sizes(color_image)
                elif key == ord('r'):  # R키: 통계 리셋
                    self.detection_count = 0
                    self.total_frames = 0
                    print("통계 리셋됨")
                elif key == ord('c'):  # C키: 현재 프레임 저장
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"charuco_test_{timestamp}.jpg"
                    cv2.imwrite(filename, color_image)
                    print(f"이미지 저장됨: {filename}")
                elif key == ord('g'):  # G키: ChArUco 보드 이미지 생성
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"charuco_board_{timestamp}.png"
                    self.generate_charuco_board_image(filename)
        
        except Exception as e:
            print(f"실시간 검출 중 오류 발생: {e}")
        
        finally:
            self.pipe.stop()
            cv2.destroyAllWindows()
            print(f"\n=== 검출 통계 ===")
            print(f"총 프레임: {self.total_frames}")
            print(f"검출 성공: {self.detection_count}")
            print(f"검출률: {(self.detection_count / self.total_frames * 100) if self.total_frames > 0 else 0:.1f}%")
    
    def test_other_sizes(self, image: np.ndarray):
        """
        다른 ChArUco 크기로 테스트합니다.
        """
        print("\n다른 ChArUco 크기 테스트 중...")
        found_sizes = self.test_different_sizes(image)
        
        if found_sizes:
            print("발견된 ChArUco 크기들:")
            for size in found_sizes:
                print(f"  - {size[0]}x{size[1]} 사각형")
            
            # 발견된 첫 번째 크기로 변경
            new_size = found_sizes[0]
            print(f"ChArUco 크기를 {new_size[0]}x{new_size[1]}로 변경합니다.")
            self.squares_x, self.squares_y = new_size
            self.charuco_board = cv2.aruco.CharucoBoard(
                size=new_size,
                squareLength=self.square_length,
                markerLength=self.marker_length,
                dictionary=self.aruco_dict
            )
        else:
            print("다른 크기로도 검출되지 않았습니다.")


def main():
    """
    메인 함수
    """
    print("ChArUco 검출 실시간 테스트")
    print("=" * 50)
    
    # ChArUco 검출기 초기화
    detector = ChArUcoDetector(
        squares_x=8,
        squares_y=11,
        square_length=23.0,  # mm
        marker_length=17.0,  # mm (이미지와 정확히 일치)
        camera_intrinsics_path="/home/ros/llm_robot/data/Calibration/Eye-in-Hand6/intrinsics/2025-07-28_20-56-40.json"
    )       
    
    try:
        # 실시간 검출 시작
        detector.run_realtime_detection()
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")


if __name__ == "__main__":
    main()