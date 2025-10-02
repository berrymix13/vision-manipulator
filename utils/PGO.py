"""
PGO (Pose Graph Optimization) with MAD Filtering

초기 변환행렬을 입력으로 받아서 MAD 필터링과 PGO 알고리즘으로 최적화하는 코드
- Eye-to-Hand 캘리브레이션을 위한 PGO 최적화
- MAD 필터링으로 outlier 제거
- 최종적으로 최적화된 cam2base 변환행렬 생성
"""

import os
import json
import numpy as np
from glob import glob
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# =============================================================================
# SE(3) 변환 유틸리티 함수들
# =============================================================================

def se3_mul(Ta, Tb):
    """SE(3) 변환행렬 곱셈: Ta * Tb"""
    Ra, ta = Ta[:3, :3], Ta[:3, 3]
    Rb, tb = Tb[:3, :3], Tb[:3, 3]
    T = np.eye(4)
    T[:3, :3] = Ra @ Rb
    T[:3, 3] = Ra @ tb + ta
    return T

def se3_inv(T):
    """SE(3) 변환행렬 역행렬"""
    R, t = T[:3, :3], T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv
    
def se3_from_Rt(R, t):
    """회전행렬과 평행이동벡터로부터 SE(3) 변환행렬 생성"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T

def log_SO3(R):
    """SO(3) 회전행렬을 로그 매핑 (rotation vector)"""
    return Rotation.from_matrix(R).as_rotvec()

def log_se3(T):
    """SE(3) 변환행렬을 로그 매핑 (rotation vector + translation)"""
    r = log_SO3(T[:3, :3])
    t = T[:3, 3]
    return np.hstack([r, t])

# =============================================================================
# PGO 최적화 함수
# =============================================================================

def optimize_cam2base_PGO(A_list, B_list, R_cam2base_init=np.eye(3), 
                         t_cam2base_init=np.zeros(3), max_nfev=200):
    """
    PGO를 사용한 cam2base 변환 최적화
    
    Args:
        A_list: base2ee 변환행렬 리스트
        B_list: target2cam 변환행렬 리스트  
        R_cam2base_init: 초기 회전행렬
        t_cam2base_init: 초기 평행이동벡터
        max_nfev: 최대 함수 평가 횟수
        
    Returns:
        X_opt: 최적화된 cam2base 변환행렬
        Y_opt: 최적화된 target2base 변환행렬
        result: 최적화 결과 정보
    """
    # 변수: X=cam2base(6), Y=target2base(6) → 총 12차원
    x0 = np.zeros(12)
    x0[:3] = Rotation.from_matrix(R_cam2base_init).as_rotvec()
    x0[3:6] = t_cam2base_init
    
    # Y 초기값 설정: A_0 * X * B_0 (합리적 초기값)
    X0 = se3_from_Rt(R_cam2base_init, t_cam2base_init)
    Y0 = se3_mul(se3_mul(A_list[0], X0), B_list[0])
    x0[6:9] = Rotation.from_matrix(Y0[:3,:3]).as_rotvec()
    x0[9:12] = Y0[:3,3]
    
    def unpack(x):
        """12차원 벡터를 X, Y 변환행렬로 변환"""
        rX, tX = x[:3], x[3:6]
        rY, tY = x[6:9], x[9:12]
        X = se3_from_Rt(Rotation.from_rotvec(rX).as_matrix(), tX)
        Y = se3_from_Rt(Rotation.from_rotvec(rY).as_matrix(), tY)
        return X, Y

    # def residual(x):
        # """잔차 함수: Y^(-1) * A * X * B = I가 되도록 최적화"""
        # X, Y = unpack(x)
        # Yin = se3_inv(Y)
        # res = []
        # for A, B in zip(A_list, B_list):
        #     T = se3_mul(se3_mul(Yin, A), se3_mul(X, B))
        #     e = log_se3(T)
        #     res.append(e)
        # return np.concatenate(res)
        
    def residual(x):
        """잔차 함수: X^{-1} A Y B = I가 되도록 최적화"""
        # 최적화 대상 변수 벡터 x를 변환행렬 X, Y로 복원
        X, Y = unpack(x)

        # X의 역행렬 (X^{-1}) 계산
        Xinv = se3_inv(X)

        res = []
        # 각 관측쌍 (A, B)에 대해 잔차를 계산
        # A = base → ee, B = target → cam
        for A, B in zip(A_list, B_list):
            # 잔차 대상 변환식:
            #   X^{-1} * A * Y * B
            #   이 값이 항등행렬(I)에 가까워지도록 최적화
            T = se3_mul(se3_mul(se3_mul(Xinv, A), Y), B)

            # SE(3) 변환행렬을 로그맵으로 표현해 6D 오차 벡터(e) 추출
            e = log_se3(T)
            res.append(e)

        # 모든 잔차 벡터를 하나로 이어붙여 반환
        return np.concatenate(res)

    
    # 최적화 실행 (outlier가 있으면 'trf'+'soft_l1' 추천)
    result = least_squares(residual, x0, method='trf', loss='soft_l1', 
                          f_scale=1.0, max_nfev=max_nfev)
    X_opt, Y_opt = unpack(result.x)
    return X_opt, Y_opt, result

# =============================================================================
# MAD 필터링 함수들
# =============================================================================

def per_frame_residuals_SE3(X, Y, A_list, B_list):
    """프레임별 잔차 계산 (회전각도, 평행이동 거리)"""
    deg_list, mm_list = [], []
    # Tinvs = np.linalg.inv(Y)
    Xinv = np.linalg.inv(X)
    for A, B in zip(A_list, B_list):
        # T = Tinvs @ A @ X @ B
        T = Xinv @ A @ Y @ B    # <= 최적화에 사용한 것과 동일하게
        r = Rotation.from_matrix(T[:3,:3]).as_rotvec()
        t = T[:3,3]
        deg_list.append(np.linalg.norm(r) * 180 / np.pi)
        mm_list.append(np.linalg.norm(t) * 1000)
    return np.array(deg_list), np.array(mm_list)

def mad_filter(vals, k=3.5):
    """
    MAD (Median Absolute Deviation) 필터링으로 outlier 제거
    
    Args:
        vals: 필터링할 값들
        k: 임계값 (기본값 3.5)
        
    Returns:
        keep: 유지할 인덱스들
        med: 중앙값
        mad: MAD 값
    """
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) + 1e-9
    z = 0.6745 * (vals - med) / mad
    keep = np.abs(z) < k
    return keep, med, mad

# =============================================================================
# 메인 실행 코드
# =============================================================================

def main():
    """메인 실행 함수"""
    # 경로 설정
    MAIN_PATH = "/home/ros/llm_robot/data/Calibration/Eye-to-Hand13"
    init_cam2base_path = f"{MAIN_PATH}/cam2base_table_normal_fix_v1.json"
    
    # 초기 cam2base 변환행렬 로드
    with open(init_cam2base_path, "r") as f:
        init_cam2base = json.load(f)
    
    R_cam2base_init = np.array(init_cam2base["R_cam2base"]).reshape(3, 3)
    t_cam2base_init = np.array(init_cam2base["t_cam2base"]).reshape(3, 1)
    
    # 데이터 로드
    pose_files = sorted(glob(f"{MAIN_PATH}/poses/*.json"))
    charuco_files = sorted(glob(f"{MAIN_PATH}/*_charuco.json"))
    
    R_target2cam_list = []
    t_target2cam_list = []
    R_base2ee_list = []
    t_base2ee_list = []
    
    for pose_file, charuco_file in zip(pose_files, charuco_files):
        # 로봇 포즈 데이터 로드
        with open(pose_file, "r") as f:
            pose = json.load(f)
        R_base2ee = np.array(pose["R_base2ee"]).reshape(3, 3)
        t_base2ee = np.array(pose["t_base2ee"]).reshape(3, 1)
        
        # 카메라 캘리브레이션 데이터 로드
        with open(charuco_file, "r") as f:
            charuco = json.load(f)
        r_tc = np.array(charuco["rvec_target2cam"], float).reshape(3)
        t_tc = np.array(charuco["tvec_target2cam"], float).reshape(3)
        R_tc = Rotation.from_rotvec(r_tc).as_matrix()
        
        R_target2cam_list.append(R_tc)
        t_target2cam_list.append(t_tc)
        R_base2ee_list.append(R_base2ee)
        t_base2ee_list.append(t_base2ee)
    
    N = len(R_target2cam_list)
    assert N == len(t_target2cam_list) == len(R_base2ee_list) == len(t_base2ee_list) and N >= 3
    
    # 변환행렬 리스트 생성
    A_list = [se3_from_Rt(Ra, ta) for Ra, ta in zip(R_base2ee_list, t_base2ee_list)]
    B_list = [se3_from_Rt(Ra, ta) for Ra, ta in zip(R_target2cam_list, t_target2cam_list)]

    # 1단계: 초기 PGO 최적화
    print("=== 1단계: 초기 PGO 최적화 ===")
    X_opt, Y_opt, info_opt = optimize_cam2base_PGO(
        A_list, B_list,
        R_cam2base_init=R_cam2base_init,
        t_cam2base_init=np.zeros(3),
        max_nfev=400
    )
    print(f"초기 최적화 완료 - t: {X_opt[:3,3]}, cost: {info_opt.cost:.6f}, nfev: {info_opt.nfev}")
    
    # 2단계: MAD 필터링으로 outlier 제거
    print("\n=== 2단계: MAD 필터링 ===")
    deg, mm = per_frame_residuals_SE3(X_opt, Y_opt, A_list, B_list)
    idx = np.arange(len(deg))
    
    # 회전/평행이동 각각 MAD 필터 적용
    keep_r, r_med, r_mad = mad_filter(deg, k=3.5)
    keep_t, t_med, t_mad = mad_filter(mm, k=3.5)
    keep = keep_r & keep_t
    
    print(f"회전 잔차 - 중앙값: {r_med:.2f}°, MAD: {r_mad:.2f}°")
    print(f"평행이동 잔차 - 중앙값: {t_med:.1f}mm, MAD: {t_mad:.1f}mm")
    print(f"전체 {len(deg)}개 중 {keep.sum()}개 inlier 선택")
    
    # 3단계: inlier만으로 재최적화
    print("\n=== 3단계: Inlier 재최적화 ===")
    A_in = [A_list[i] for i in idx[keep]]
    B_in = [B_list[i] for i in idx[keep]]
    
    X_in, Y_in, info_in = optimize_cam2base_PGO(
        A_in, B_in,
        R_cam2base_init=np.eye(3), 
        t_cam2base_init=np.zeros(3),
        max_nfev=2000
    )
    
    # 최종 결과 출력
    deg_in, mm_in = per_frame_residuals_SE3(X_in, Y_in, A_in, B_in)
    print(f"최종 회전 잔차 - 평균: {deg_in.mean():.2f}°, 최대: {deg_in.max():.2f}°")
    print(f"최종 평행이동 잔차 - 평균: {mm_in.mean():.1f}mm, 최대: {mm_in.max():.1f}mm")
    print(f"최종 t_cam2base: {X_in[:3,3]}")
    
    # 결과 저장
    result = {
        "R_cam2base": X_in[:3,:3].reshape(3,3).tolist(),
        "t_cam2base": X_in[:3,3].flatten().tolist()
    }
    
    output_path = f"{MAIN_PATH}/cam2base_pgo_inliers.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n결과 저장 완료: {output_path}")

if __name__ == "__main__":
    main()