import os

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from scipy.interpolate import make_smoothing_spline, UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from sklearn.kernel_ridge import KernelRidge

from pcd_processing import process_frame, merge_point_clouds
from pcd_slicing import slice_and_project
from processing_helper import fit_plane_to_point_cloud, visualize_plane
from src.extract_xlsx import extract_xlsx
from src.pcd_slicing import slice_point_cloud, rotate_points
from src.processing_helper import pick_direction_interactively


def process_point_cloud(
        bag_file_floor, bag_file_water, output_folder, eps=0.01, min_points=50, max_distance=1,
        frame_numbers=[5, 6000], frame_count=20, load=False,
        step_size=0.001, num_slices=20, plot_ultrasound=False):
    if load:
        clustered_first = o3d.io.read_point_cloud(f"{output_folder}/output_point_cloud_first.ply")
        clustered_second = o3d.io.read_point_cloud(f"{output_folder}/output_point_cloud_second.ply")
    else:
        clustered_first = process_frame(bag_file_floor, frame_numbers[0], frame_count, max_distance, eps, min_points,
                                          output_folder)
        clustered_second = process_frame(bag_file_water, frame_numbers[1], frame_count, max_distance, eps, min_points,
                                        output_folder)

    o3d.io.write_point_cloud(f"{output_folder}/output_point_cloud_first.ply", clustered_first)
    o3d.io.write_point_cloud(f"{output_folder}/output_point_cloud_second.ply", clustered_second)

    direction, p1, p2 = pick_direction_interactively(clustered_first)
    normal = fit_plane_to_point_cloud(clustered_first)
    visualize_plane(clustered_first, p1, p2, normal)

    # Compute shift direction using cross product
    slice_direction = np.cross(normal, (p2 - p1))
    slice_direction /= np.linalg.norm(slice_direction)  # Normalize

    all_projections_first = []
    all_projections_second = []

    # Generate shifts symmetrically around the center (negative and positive steps)
    shift_values = np.linspace(-step_size * (num_slices // 2), step_size * (num_slices // 2), num_slices)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    rotation_matrix = None
    fitted_line_final = None
    shift_final = None

    for i, shift in enumerate(shift_values):
        offset = shift * slice_direction
        p1_shifted, p2_shifted = p1 + offset, p2 + offset
        #rotated_points_2d = slice_and_project(clustered_pcd, p1_shifted, p2_shifted, normal, 0.01)
        projected_points_first, corrected_normal_first, sliced_colors_first = slice_point_cloud(clustered_first, p1_shifted, p2_shifted, normal, 0.01)
        projected_points_second, corrected_normal_second, sliced_colors_second = slice_point_cloud(clustered_second, p1_shifted, p2_shifted,
                                                                              normal, 0.01)
        rotated_points_2d_first, rotation_matrix, fitted_line, shift = rotate_points(projected_points_first, corrected_normal_first, rotation_matrix, shift_final)
        if not fitted_line_final:
            fitted_line_final = fitted_line
        if not shift_final:
            shift_final = shift
        rotated_points_2d_second, rotation_matrix, _, _ = rotate_points(projected_points_second, corrected_normal_second, rotation_matrix, shift_final)
        all_projections_first.append(rotated_points_2d_first)
        all_projections_second.append(rotated_points_2d_second)

        # Save individual slice plot
        plt.figure(figsize=(8, 6))
        plt.scatter(rotated_points_2d_second[:, 0], rotated_points_2d_second[:, 1] * 1000, s=1, alpha=0.5, label=f'Slice {i + 1}')
        plt.xlabel("X")
        plt.ylabel("Y (mm)")
        plt.title(f"2D Projection - Slice {i + 1}")
        plt.legend()
        plt.savefig(f"{output_folder}/slice_{i + 1}.png")
        plt.close()

    # Extract external data
    xlsx_data = extract_xlsx()

    # Compute the average and standard deviation with variable-length handling
    min_length = min(len(proj) for proj in all_projections_second)
    trimmed_projections = [proj[:min_length] for proj in all_projections_second]  # Trim to the shortest projection
    avg_projection = np.mean(trimmed_projections, axis=0)
    std_dev_projection = np.std(trimmed_projections, axis=0)

    # Plot the final comparison
    plt.figure(figsize=(8, 6))

    # Plot individual projections
    for i, projection in enumerate(all_projections_second):
        plt.scatter(projection[:, 0], projection[:, 1] * 1000, s=1, alpha=0.3)
    for j, projection_first in enumerate(all_projections_first):
        plt.scatter(projection_first[:, 0], projection_first[:, 1] * 1000, s=1, alpha=0.3)
    if fitted_line_final:
        x_fit, y_fit = fitted_line_final
        #plt.plot(x_fit, y_fit, color='red')

    # Plot average projection
    #plt.scatter(-avg_projection[:, 0], avg_projection[:, 1] * 1000 + 27, s=3, color='red')
    x = avg_projection[:, 0]
    y = avg_projection[:, 1] * 1000

    poly = np.poly1d(np.polyfit(x, y, 15))
    #plt.scatter(x, y, s=3, color='red', label='Raw Data')
    plt.plot(x, poly(x), color='red', label='Smoothed Mean')

    # Plot standard deviation as a shaded region
    #plt.fill_between(
    #    -avg_projection[:, 0],
    #    (avg_projection[:, 1] - std_dev_projection[:, 1]) * 1000,
    #    (avg_projection[:, 1] + std_dev_projection[:, 1]) * 1000,
    #    color='red', alpha=0.2, label='Standard Deviation'
    #)

    # Optionally plot ultrasound depth
    if plot_ultrasound:
        plt.plot(xlsx_data['Strecke'], xlsx_data['Wassertiefe'], color='b', marker='o', linestyle='-', markersize=1,
                 label='Ultraschall Wassertiefe')

    # Add labels, title, and legend
    plt.xlabel("X")
    plt.ylabel("Y (mm)")
    plt.title("Vergleich RealSense und Ultraschall Wassertiefe")
    plt.legend()

    # Save the final comparison plot
    plt.savefig(f"{output_folder}/overlay_plot.png")
    plt.show()


bag_file_water = "../data_experiments/20250404_100521.bag" #135 for floaters
bag_file_floor = "../data_experiments/20250404_094437.bag"
output_folder = "../outputs"
process_point_cloud(bag_file_floor, bag_file_water, output_folder, frame_numbers=[10, 135], frame_count=1, load=False, plot_ultrasound=False)
