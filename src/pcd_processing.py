import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

from src.bag_file_handling import read_bag_file


def depth_to_point_cloud(depth_image, color_image, depth_scale, intrinsics, max_distance=5.0):
    h, w = depth_image.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))

    z = depth_image * depth_scale
    x = (j - intrinsics.ppx) * z / intrinsics.fx
    y = (i - intrinsics.ppy) * z / intrinsics.fy

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = color_image.reshape(-1, 3) / 255.0  # Normalize RGB values

    valid = (points[:, 2] > 0) & (points[:, 2] < max_distance)  # Filter out invalid and far points

    return points[valid], colors[valid]

def sor_filter(pcd, nb_neighbors=5000, std_ratio=1.1):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)

def create_open3d_point_cloud(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def cluster_pcd(pcd, eps=0.5, min_points=10):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)  # Get the color information from the original point cloud

    # Perform DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_points, n_jobs=-1, algorithm='ball_tree')
    labels = db.fit_predict(points)
    max_label = labels.max()
    print(f"Point cloud has {max_label + 1} clusters")

    # Find the largest cluster
    unique_labels = set(labels)
    largest_cluster_label = None
    largest_cluster_size = 0

    for label in unique_labels:
        if label == -1:
            continue
        cluster_size = np.sum(labels == label)
        if cluster_size > largest_cluster_size:
            largest_cluster_size = cluster_size
            largest_cluster_label = label

    largest_cluster_points = points[labels == largest_cluster_label]
    largest_cluster_colors = colors[labels == largest_cluster_label]

    largest_cluster_pcd = create_open3d_point_cloud(largest_cluster_points, largest_cluster_colors)

    return largest_cluster_pcd

def process_frame(bag_file, frame_number, frame_count, max_distance, eps, min_points, output_folder):
    # Step 1: Read bag file and extract depth and color images
    depth_image, color_image, depth_scale, intrinsics = read_bag_file(
        bag_file, start_frame=frame_number, frame_count=frame_count
    )

    depth_normalized = depth_image.astype(np.float32) * depth_scale
    depth_normalized = (depth_normalized - np.min(depth_normalized)) / (
                np.max(depth_normalized) - np.min(depth_normalized))
    depth_normalized = np.clip(depth_normalized, 0.0, 1.0)
    colormap = plt.get_cmap('jet')
    colored_depth = colormap(1.0 - depth_normalized)[:, :, :3]
    colored_depth_8bit = (colored_depth * 255).astype(np.uint8)
    colored_depth_bgr = cv2.cvtColor(colored_depth_8bit, cv2.COLOR_RGB2BGR)
    cv2.imwrite('depth_colored.png', colored_depth_bgr)

    # Step 2: Convert depth image to point cloud
    points, colors = depth_to_point_cloud(
        depth_image, color_image, depth_scale, intrinsics, max_distance=max_distance
    )

    # Step 3: Create an Open3D point cloud
    pcd = create_open3d_point_cloud(points, colors)
    o3d.io.write_point_cloud("raw_pcd.ply", pcd)
    print(f"Frame {frame_number}: The point cloud contains {len(pcd.points)} points.")

    # Step 4: Cluster the point cloud and save the largest cluster as a PLY file
    clustered_pcd = cluster_pcd(pcd, eps=eps, min_points=min_points)
    cl, ind = clustered_pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=1.5)
    clustered_pcd = clustered_pcd.select_by_index(ind)

    return clustered_pcd

def merge_point_clouds(pcd_list):
    # Initialize an empty point cloud
    merged_pcd = o3d.geometry.PointCloud()

    # Merge all point clouds in the list
    for pcd in pcd_list:
        merged_pcd += pcd

    return merged_pcd