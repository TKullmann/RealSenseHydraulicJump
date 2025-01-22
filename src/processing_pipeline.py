import sys

import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from bag_file_handling import read_bag_file
from pcd_processing import depth_to_point_cloud, create_open3d_point_cloud, save_clusters_as_ply
from pcd_slicing import slice_point_cloud_rotated, extract_thin_slice
from processing_helper import rotation_matrix_from_axis_angle, refine_flow_direction
from src.processing_helper import pick_direction_interactively


def process_point_cloud(bag_file, output_folder, eps=0.01, min_points=50, max_distance=0.9, frame_number=5, frame_count=20, n_slices=10):
    ###################################################################### Single Frame Code Below ######################################################################
    # Step 1: Read the bag file and extract depth and color images
    depth_image, color_image, depth_scale, intrinsics = read_bag_file(bag_file, start_frame=frame_number, frame_count=frame_count)

    # Step 2: Convert depth image to point cloud
    points, colors = depth_to_point_cloud(depth_image, color_image, depth_scale, intrinsics, max_distance=max_distance)

    # Step 3: Create an Open3D point cloud
    pcd = create_open3d_point_cloud(points, colors)
    print(f"The point cloud contains {len(pcd.points)} points.")
    ###################################################################### Single Frame Code Above ######################################################################


    # Step 4: Cluster the point cloud and save the largest cluster as a PLY file
    clustered_pcd = save_clusters_as_ply(pcd, eps=eps, min_points=min_points, output_folder=output_folder)
    cl, ind = clustered_pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=1.5)
    clustered_pcd = clustered_pcd.select_by_index(ind)

    direction, p1, p2 = pick_direction_interactively(clustered_pcd)

    sys.exit("Halt Stopp!")

    # Step 5: Write the clustered point cloud to a file
    o3d.io.write_point_cloud(f"{output_folder}/output_point_cloud.ply", clustered_pcd)

    # Step 6: Refine the flow direction using PCA
    flow_direction = refine_flow_direction(clustered_pcd)
    flow_direction_normalized = flow_direction / np.linalg.norm(flow_direction)

    # Step 7: Define a point on the plane (mean of the point cloud)
    point_on_plane = np.mean(np.asarray(clustered_pcd.points), axis=0)

    # Step 8: Slice the point cloud with the rotated plane
    ############################################### Single Slice #######################################################
    slice = extract_thin_slice(clustered_pcd, flow_direction_normalized, point_on_plane, thickness=0.3)
    
    if not slice.is_empty():
        o3d.io.write_point_cloud(f"{output_folder}/slice.ply", slice)
        print("Saved slice to slice.ply")
    ############################################### Single Slice #######################################################

    ############################################### Multiple Slices ####################################################
    """all_slices = []
    slice_offset = 0.1
    for i in range(n_slices):
        perpendicular_direction = np.cross(flow_direction_normalized, np.array([1, 0, 0]))

        if np.linalg.norm(perpendicular_direction) == 0:
            perpendicular_direction = np.cross(flow_direction_normalized, np.array([0, 1, 0]))

        perpendicular_direction_normalized = perpendicular_direction / np.linalg.norm(perpendicular_direction)

        shifted_point_on_plane = point_on_plane + perpendicular_direction_normalized * slice_offset * i

        # Extract the slice with the fixed thickness
        slice = extract_thin_slice(clustered_pcd, flow_direction_normalized, shifted_point_on_plane)

        if not slice.is_empty():
            all_slices.append(slice)
            o3d.io.write_point_cloud(f"{output_folder}/slice_{i}.ply", slice)
            print(f"Saved slice_{i} to slice_{i}.ply")"""
    ############################################### Multiple Slices ####################################################


    # Step 10: Fit a plane to the slice points using PCA
    points = np.asarray(slice.points)
    pca = PCA(n_components=3)
    pca.fit(points)
    normal_vector = pca.components_[-1]  # The normal vector is the last component

    # Step 11: Compute the rotation matrix to align the normal vector with the Z-axis
    z_axis = np.array([0, 0, 1])
    axis = np.cross(normal_vector, z_axis)
    axis_length = np.linalg.norm(axis)

    if axis_length > 0:
        axis = axis / axis_length
        angle = np.arccos(np.dot(normal_vector, z_axis))
        rotation_matrix = rotation_matrix_from_axis_angle(axis, angle)
    else:
        rotation_matrix = np.eye(3)  # The normal vector is already aligned

    rotated_points = np.dot(points - np.mean(points, axis=0), rotation_matrix.T)

    # Step 12: Perform 2D projection and linear regression
    projected_points = rotated_points[:, :2]
    projected_points[:, [0, 1]] = projected_points[:, [1, 0]]  # Swap x and y coordinates
    third_points = projected_points[:len(points) // 3]

    regressor = LinearRegression()
    regressor.fit(third_points[:, 0].reshape(-1, 1), third_points[:, 1])

    slope = regressor.coef_[0]
    angle = np.arctan(slope)  # Angle in radians

    rotation_matrix_2d = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle), np.cos(-angle)]
    ])

    rotated_points_2d = np.dot(projected_points - np.mean(projected_points, axis=0), rotation_matrix_2d.T)

    # Step 13: Visualize the 2D projection
    plt.figure(figsize=(8, 6))
    plt.scatter(rotated_points_2d[:, 0], rotated_points_2d[:, 1], s=1)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Projection of Point Cloud")
    plt.axis("equal")
    plt.savefig(f"{output_folder}/projection.png")


bag_file = "../dataPraktikumsrinneLong/angle_full_cycle.bag"
output_folder = "../outputs"
process_point_cloud(bag_file, output_folder, frame_number=0, frame_count=500, n_slices=10)
