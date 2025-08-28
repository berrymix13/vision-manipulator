import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation

def create_roi_pcd_from_depth(depth_path, yolo_outputs, camera_matrix):
    depth = np.load(depth_path)
    x1, y1, x2, y2 = yolo_outputs[0]["bbox"]
    roi_depth = depth[y1:y2, x1:x2]
    
    fx = camera_matrix[0][0]    # 수평 초점 거리
    fy = camera_matrix[1][1]    # 수직 초점 거리
    ppx = camera_matrix[0][2]    # center x
    ppy = camera_matrix[1][2]    # center y
    
    roi_ppx = ppx - x1
    roi_ppy = ppy - y1

    roi_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=roi_depth.shape[1],
        height=roi_depth.shape[0],
        fx=fx,
        fy=fy,
        cx=roi_ppx,
        cy=roi_ppy
    )

    depth_contig = np.ascontiguousarray(roi_depth)
    depth_o3d = o3d.geometry.Image(depth_contig)

    dummy_gray = np.ones_like(roi_depth, dtype=np.uint8) * 128
    dummy_rgb = np.stack([dummy_gray]*3, axis=2) 
    dummy_color = o3d.geometry.Image(dummy_rgb)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        dummy_color,
        depth_o3d,
        depth_scale=1000,
        depth_trunc = 2.0,
        convert_rgb_to_intensity = False    
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        roi_intrinsic
    )
    _, inliers = pcd.segment_plane(
        distance_threshold=0.005,
        ransac_n=3,
        num_iterations=100
    )

    pcd = pcd.select_by_index(inliers)
    return pcd


def calculate_rpy(pcd):    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.normalize_normals()    
        
    normals = np.asarray(pcd.normals)         # (N,3) 배열
    avg_normal = normals.mean(axis=0)         # 평균 법선 벡터
    avg_normal /= np.linalg.norm(avg_normal)  # 단위 벡터로 정규화

    # PCD의 중심점(centroid) 계산
    points = np.asarray(pcd.points)
    centroid = points.mean(axis=0)

    z_axis = np.array([0, 0, 1])
    # 회전축과 회전각을 사용하여 회전 행렬을 계산합니다.
    axis = np.cross(z_axis, avg_normal)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        rot_mat_normal = np.eye(3)
    else:
        axis /= axis_norm
        angle = np.arccos(np.dot(z_axis, avg_normal))
        rot_mat_normal = Rotation.from_rotvec(axis * angle).as_matrix()
        
    rot_mat_eul = Rotation.from_matrix(rot_mat_normal)
    eul = rot_mat_eul.as_euler('zyx', degrees=True) # Z-Y-X (Yaw-Pitch-Roll)
    pitch_normal, roll_normal = eul[1], eul[2]

    z_axis = np.array([0, 0, 1])
    # 평균 법선 방향 계산 
    points_centered = points - centroid 
    cov = np.cov(points_centered.T)             # 3×3
    _, vecs = np.linalg.eigh(cov)           # 고유값 오름차순
    # vecs[:, -1] 은 장축, vecs[:, -2] 는 단축(minor axis)
    major_axis = vecs[:, -1]   # 최대 분산을 갖는 축 = 그리퍼 개방 방향
    minor_axis = vecs[:, -2]   # 중간 분산을 갖는 축
    yaw_final = np.degrees(np.arctan2(np.dot(np.cross(z_axis, major_axis), minor_axis), np.dot(z_axis, major_axis)))
    return roll_normal, pitch_normal, yaw_final