import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
from sklearn.linear_model import LinearRegression

from src.processing_helper import align_to_xy_plane


def slice_point_cloud_rotated(pcd, flow_direction, point_on_plane, keep_positive=True, shift_distance = 0.05):
    flow_direction = flow_direction / np.linalg.norm(flow_direction)

    arbitrary_vector = np.array([1, 0, 0]) if not np.allclose(flow_direction, [1, 0, 0]) else np.array([0, 1, 0])
    perp_vector = np.cross(flow_direction, arbitrary_vector)
    perp_vector = perp_vector / np.linalg.norm(perp_vector)

    rotated_normal = np.cross(flow_direction, perp_vector)

    point_on_plane = point_on_plane + rotated_normal * shift_distance

    points = np.asarray(pcd.points)
    distances = np.dot(points - point_on_plane, rotated_normal)

    if keep_positive:
        mask = distances >= 0
    else:
        mask = distances < 0

    sliced_pcd = pcd.select_by_index(np.where(mask)[0])
    return sliced_pcd


def extract_thin_slice(pcd, flow_direction, point_on_plane, thickness=0.1):
    flow_direction = flow_direction / np.linalg.norm(flow_direction)

    arbitrary_vector = np.array([1, 0, 0]) if not np.allclose(flow_direction, [1, 0, 0]) else np.array([0, 1, 0])
    perp_vector = np.cross(flow_direction, arbitrary_vector)
    perp_vector = perp_vector / np.linalg.norm(perp_vector)

    rotated_normal = np.cross(flow_direction, perp_vector)

    points = np.asarray(pcd.points)
    distances = np.dot(points - point_on_plane, rotated_normal)

    mask = np.abs(distances) <= (thickness / 2)

    sliced_pcd = pcd.select_by_index(np.where(mask)[0])
    return sliced_pcd

def slice_and_project(pcd, p1, p2, normal, slice_width):
    # Convert points and normal to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)
    normal = np.array(normal)

    # Extract points from the point cloud
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Calculate the direction vector between p1 and p2
    direction_vector = p2 - p1
    direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize

    normal = np.cross(direction_vector, normal)
    normal = normal / np.linalg.norm(normal)

    # Calculate the perpendicular distance from each point to the plane
    distances = np.abs(np.dot(points - p1, normal)) / np.linalg.norm(normal)

    # Filter points within the slice width
    slice_mask = distances <= slice_width
    sliced_points = points[slice_mask]
    sliced_colors = colors[slice_mask]

    # Project points onto the plane
    projected_points = []
    for point in sliced_points:
        # Compute the projection of the point onto the plane
        projection = point - np.dot(point - p1, normal) * normal / np.linalg.norm(normal) ** 2
        projected_points.append(projection)

    # Convert to Open3D point cloud
    projected_points = np.array(projected_points)

    # Align projected points to the XY plane
    aligned_points = align_to_xy_plane(projected_points, normal)

    # Visualize the points in 2D with Matplotlib
    """plt.figure(figsize=(8, 6))
    plt.scatter(aligned_points[:, 0], aligned_points[:, 1], s=1, c=sliced_colors, alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Visualization of Projected Points with Colors')
    plt.axis('equal')
    plt.show()"""

    aligned_points = aligned_points[np.lexsort((aligned_points[:, 1], aligned_points[:, 0]))]
    aligned_points = aligned_points[:, :2]
    total_points = len(aligned_points)
    left_cutoff = int(total_points * 0.8)
    lowest_points = aligned_points[left_cutoff:]

    """plt.figure(figsize=(8, 6))
    plt.scatter(lowest_points[:, 0], lowest_points[:, 1], s=1, alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Visualization of Lowest Points')
    plt.axis('equal')
    plt.show()"""

    x = lowest_points[:, 0].reshape(-1, 1)
    y = lowest_points[:, 1]

    # Fit a line
    model = LinearRegression()
    model.fit(x, y)

    slope = model.coef_[0]
    intercept = model.intercept_
    angle = np.arctan(slope)

    rotation_matrix = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle), np.cos(-angle)]
    ])

    # Rotate all points
    rotated_points = np.dot(aligned_points, rotation_matrix.T)
    rotated_points[:, 1] -= np.min(rotated_points[:, 1])

    return rotated_points


def slice_point_cloud(pcd, p1, p2, normal, slice_width):
    """Slices the point cloud and projects the points onto a plane."""

    p1, p2, normal = map(np.array, (p1, p2, normal))

    # Extract points from the point cloud
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Calculate the direction vector and correct the normal
    direction_vector = p2 - p1
    direction_vector /= np.linalg.norm(direction_vector)

    corrected_normal = np.cross(direction_vector, normal)
    corrected_normal /= np.linalg.norm(corrected_normal)

    # Compute perpendicular distances
    distances = np.abs(np.dot(points - p1, corrected_normal)) / np.linalg.norm(corrected_normal)

    # Filter points within slice width
    slice_mask = distances <= slice_width
    sliced_points = points[slice_mask]
    sliced_colors = colors[slice_mask]

    # Project points onto the plane
    projected_points = sliced_points - np.dot(sliced_points - p1, corrected_normal)[:, None] * corrected_normal

    return projected_points, corrected_normal, sliced_colors


def rotate_points(projected_points, normal, rotation_matrix=None, shift=None):
    """Rotates projected points using either an existing rotation matrix or fitting a line."""

    # Align projected points to the XY plane
    aligned_points = align_to_xy_plane(projected_points, normal)

    # Keep only x and y coordinates
    aligned_points = aligned_points[:, :2]

    # If no rotation matrix is provided, compute one using line fitting
    if rotation_matrix is None:
        # Fit a line using all points
        x = aligned_points[:, 0].reshape(-1, 1)
        y = aligned_points[:, 1]
        model = LinearRegression().fit(x, y)

        # Compute rotation matrix
        angle = np.arctan(model.coef_[0])
        rotation_matrix = np.array([
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle), np.cos(-angle)]
        ])

        # Generate fitted line points for visualization
        x_fit = np.linspace(np.min(x), np.max(x), 100)
        y_fit = model.predict(x_fit.reshape(-1, 1))
        fitted_line = (x_fit, y_fit)
    else:
        fitted_line = None  # No line fitting if rotation matrix is provided

    # Apply the rotation matrix to all points
    rotated_points = np.dot(aligned_points, rotation_matrix.T)

    if shift is None:
        # Compute y-values of the fitted line in the rotated space
        fitted_y_values = model.predict(aligned_points[:, 0].reshape(-1, 1))
        transformed_fitted_y = np.dot(np.column_stack((aligned_points[:, 0], fitted_y_values)), rotation_matrix.T)[:, 1]

        # Compute shift value (mean of transformed fitted y-values)
        shift = np.mean(transformed_fitted_y)

    # Apply shift to align the fitted line with the x-axis
    rotated_points[:, 1] -= shift

    return rotated_points, rotation_matrix, fitted_line, shift