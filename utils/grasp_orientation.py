import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from camera import load_intrinsics

def create_pcd_from_depth(d_path, intr_path, yolo_outputs,
                          depth_scale=1000, depth_trunc=2.0):
    """_summary_

    Args:
        d_path : Path to the depth image file (.npy)
        intr_path : Path to the intrinsic file (.json)
        yolo_outputs : List of yolo outputs
        depth_scale: Scale factor for depth values
        depth_trunc: Maximum depth threshold

    Returns:
        point_cloud: Open3D PointCloud object
    """
    # 1. ROI depth 추출
    depth = np.load(d_path)
    camera_matrix, _ = load_intrinsics(intr_path)
    
    x1, y1, x2, y2 = yolo_outputs[0]["bbox"]
    depth = np.load(d_path)
    roi_depth = depth[y1:y2, x1:x2]
    
    # 2. ROI intrinsic 생성
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
    
    # 3. ROI PCD 생성
    depth_contig = np.ascontiguousarray(roi_depth)
    depth_o3d = o3d.geometry.Image(depth_contig)

    dummy_gray = np.ones_like(roi_depth, dtype=np.uint8) * 128
    dummy_rgb = np.stack([dummy_gray]*3, axis=2) 
    dummy_color = o3d.geometry.Image(dummy_rgb)
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        dummy_color,
        depth_o3d,
        depth_scale=depth_scale,
        depth_trunc = depth_trunc,
        convert_rgb_to_intensity = False
    )

    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, roi_intrinsic)

def segment_plane_from_pointcloud(pcd, distance_threshold=0.005,
                                  ransac_n=3, num_iterations=1000):
    """
    Segment plane from point cloud using RANSAC.
    
    Args:
        pcd: Input point cloud
        distance_threshold: Maximum distance from point to plane
        ransac_n: Number of points to estimate plane
        num_iterations: Number of RANSAC iterations
        
    Returns:
        segmented_pcd: Point cloud containing only inlier points
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    
    return plane_model, pcd.select_by_index(inliers)


def calculate_rotation_matrix_to_z_axis(
    normal_vector: np.ndarray
) -> np.ndarray:
    """
    Calculate rotation matrix to align normal vector with Z-axis.
    
    Args:
        normal_vector: Unit normal vector (3,)
        
    Returns:
        rotation_matrix: 3x3 rotation matrix
    """
    z_axis = np.array([0.0, 0.0, 1.0])
    
    # Calculate rotation axis (cross product)
    axis = np.cross(z_axis, normal_vector)
    axis_norm = np.linalg.norm(axis)
    
    # Handle case where vectors are parallel
    if axis_norm < 1e-6:
        return np.eye(3)
    
    # Normalize axis and calculate rotation angle
    axis /= axis_norm
    angle = np.arccos(np.dot(z_axis, normal_vector))
    
    # Create rotation matrix
    return Rotation.from_rotvec(axis * angle).as_matrix()

def calculate_gripper_yaw_from_pca(
    points_2d: np.ndarray
) -> float:
    """
    Calculate gripper yaw angle using PCA on 2D projected points.
    
    Args:
        points_2d: 2D points array (N, 2)
        
    Returns:
        yaw_angle: Yaw angle in degrees
    """
    # Calculate covariance matrix
    cov_matrix = np.cov(points_2d.T)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Minor axis (second largest eigenvalue) determines gripper opening direction
    minor_axis = eigenvectors[:, -2]
    
    # Calculate yaw angle from minor axis
    yaw_angle = np.degrees(np.arctan2(minor_axis[1], minor_axis[0]))
    
    return yaw_angle


def pcd_to_surface_normal(pcd):
    # 1. Estimate and normalize normals
    pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.normalize_normals()

    # 2. Calculate average normal direction
    normals = np.asarray(pcd.normals)         # (N,3) 배열
    avg_normal = normals.mean(axis=0)         # 평균 법선 벡터
    avg_normal /= np.linalg.norm(avg_normal)  # 단위 벡터로 정규화

    # 3. Calculate centroid
    points = np.asarray(pcd.points)
    centroid = points.mean(axis=0)
    
    # Step 4: Calculate rotation matrix to align normal with Z-axis
    rotation_matrix = calculate_rotation_matrix_to_z_axis(avg_normal)
        
    # Step 5: Transform points to centroid-centered coordinates
    points_centered = points - centroid
    
    # Step 6: Apply rotation to align normal with Z-axis
    rotation_matrix_inv = rotation_matrix.T
    points_aligned = (rotation_matrix_inv @ points_centered.T).T
    
    # Step 7: Project to XY plane
    points_2d = points_aligned[:, :2]
    
    # Step 8: Calculate gripper yaw using PCA
    yaw_angle = calculate_gripper_yaw_from_pca(points_2d)
    
    return rotation_matrix, yaw_angle
