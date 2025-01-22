import numpy as np


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