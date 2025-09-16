
"""_summary_

회전행렬만 보정하지 않고 translation 보정까지 추가.
벽면을 기준으로 보정하도록 수정.

"""

import json, numpy as np, open3d as o3d
from scipy.spatial.transform import Rotation as R
from camera import load_intrinsics
from glob import glob

# ----- 공용 유틸 -----
def to_4x4(Rm, t):
    T = np.eye(4); T[:3,:3] = Rm; T[:3,3] = np.asarray(t).reshape(3); return T

def load_cam2base(cam2base_path):
    with open(cam2base_path, "r") as f:
        d = json.load(f)
    R_cb = np.array(d["R_cam2base"], float).reshape(3,3)
    t_cb = np.array(d["t_cam2base"], float).reshape(3)
    return R_cb, t_cb

def save_cam2base(cam2base_path_out, R_cb_new, t_cb):
    data = {"R_cam2base": R_cb_new.reshape(-1).tolist(),
            "t_cam2base": t_cb.reshape(-1).tolist()}
    with open(cam2base_path_out, "w") as f:
        json.dump(data, f, indent=2)

# ----- depth→PCD (전체 장면) -----
def depth_to_pcd_full(depth_npy_path, K, depth_scale=1000.0, depth_trunc=5.0):
    D = np.load(depth_npy_path).astype(np.uint16)  # mm 기준이면 uint16 OK
    fx, fy = float(K[0,0]), float(K[1,1])
    cx, cy = float(K[0,2]), float(K[1,2])
    intr = o3d.camera.PinholeCameraIntrinsic(D.shape[1], D.shape[0], fx, fy, cx, cy)
    depth_o3d = o3d.geometry.Image(D)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d, intr, depth_scale=depth_scale, depth_trunc=depth_trunc, stride=1
    )
    return pcd  # cam 좌표계

# ----- 벽면 평면 법선(카메라 좌표계) -----
def wall_normal_cam_from_depth(depth_npy_path, K,
                              depth_scale=1000.0, depth_trunc=5.0,
                              dist_thresh=0.008, ransac_n=3, iters=2000):
    pcd = depth_to_pcd_full(depth_npy_path, K, depth_scale, depth_trunc)
    if len(pcd.points) < 1000:
        raise RuntimeError("PCD too small")

    plane, inliers = pcd.segment_plane(distance_threshold=dist_thresh,
                                       ransac_n=ransac_n, num_iterations=iters)
    a, b, c, d = plane  # ax+by+cz+d=0
    n_cam = np.array([a, b, c], float)
    n_cam /= (np.linalg.norm(n_cam) + 1e-12)

    # 벽면의 경우 카메라가 벽을 향하고 있으므로, 법선은 카메라에서 멀어지는 방향(-z)에 가깝습니다.
    # 벽면이 카메라에서 멀어지는 방향을 향하도록 정렬
    if n_cam[2] > 0:  # z 성분이 양수면 뒤집기
        n_cam = -n_cam
    
    return n_cam

# ----- 평균 법선 기반 보정 -----
def refine_cam2base_wall_tilt(cam2base_path, depth_list, K,
                             depth_scale=1000.0, depth_trunc=5.0,
                             out_path=None, wall_ref=np.array([0,0,-1.0])):
    R_cb, t_cb = load_cam2base(cam2base_path)
    
    p_cams = []
    for dp in depth_list:
        pcd = depth_to_pcd_full(dp, K, depth_scale, depth_trunc)
        if len(pcd.points) < 1000:
            continue
        plane, inliers = pcd.segment_plane(distance_threshold=0.008,
                                           ransac_n=3, num_iterations=2000)
        pcd_inliers = pcd.select_by_index(inliers)
        if len(pcd_inliers.points) > 0:
            p_cams.append(np.mean(np.asarray(pcd_inliers.points), axis=0))

    if not p_cams:
        raise RuntimeError("No wall plane found in any depth image.")

    n_bases = []
    for dp in depth_list:
        n_cam = wall_normal_cam_from_depth(dp, K, depth_scale, depth_trunc)
        n_base = R_cb @ n_cam                       # 회전만 적용
        # 벽면 법선 방향으로 정렬(평균에 기여 방향 통일)
        if np.dot(n_base, wall_ref) < 0:
            n_base = -n_base
        n_bases.append(n_base)

    nbar = np.mean(np.stack(n_bases, axis=0), axis=0)
    nbar /= (np.linalg.norm(nbar) + 1e-12)

    # 벽면 법선을 z축 반대 방향으로 정렬
    z_wall = np.array([0.0, 0.0, -1.0])
    axis = np.cross(nbar, z_wall)
    s = np.linalg.norm(axis)
    if s < 1e-8:
        dR = np.eye(3)  # 이미 정렬됨
    else:
        axis /= s
        angle = np.arctan2(s, np.dot(nbar, z_wall))
        dR = R.from_rotvec(axis * angle).as_matrix()

    R_cb_new = dR @ R_cb                 
    # t_cb_new = dR @ t_cb
    p_cam_mean = np.mean(np.stack(p_cams, axis=0), axis=0) 
    p_base_old = R_cb @ p_cam_mean + t_cb
    t_new = p_base_old - R_cb_new @ p_cam_mean

    if out_path:                         # 저장 원하면
        save_cam2base(out_path, R_cb_new, t_new)

    # 디버그 정보
    before_angle = np.degrees(np.arccos(np.clip(np.dot(nbar, z_wall), -1, 1)))
    after_nbar = (dR @ nbar)
    after_angle = np.degrees(np.arccos(np.clip(np.dot(after_nbar, z_wall), -1, 1)))

    info = {
        "nbar_before_base": nbar.tolist(),
        "misalign_deg_before": float(before_angle),
        "nbar_after_base": after_nbar.tolist(),
        "misalign_deg_after": float(after_angle),
        "R_cam2base_new": R_cb_new
    }
    return R_cb_new, t_cb, info


# 준비물
MAIN_DIR = "/home/ros/llm_robot/data//Calibration/Eye-to-Hand11"
cam2base_path  = f"{MAIN_DIR}/cam2base_icp_point.json"
depth_list = sorted(glob(f"{MAIN_DIR}/depth/*.npy"))[-10:]

intr_path = f"{MAIN_DIR}/intrinsics/2025-08-27_10-05-33.json"
camera_matrix, dist_coeffs = load_intrinsics(intr_path)  # 이미 보유한 함수

# 실행
R_cb_new, t_cb, info = refine_cam2base_wall_tilt(
    cam2base_path, depth_list, camera_matrix,  # K= camera_matrix
    depth_scale=1000.0, depth_trunc=5.0,
    out_path=f"{MAIN_DIR}/cam2base_wall_normal_fix.json"
)
print("misalign(before, after) [deg]:", info["misalign_deg_before"], info["misalign_deg_after"])
