import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN


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

def save_clusters_as_ply(pcd, eps=0.5, min_points=10, output_folder="clusters"):
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