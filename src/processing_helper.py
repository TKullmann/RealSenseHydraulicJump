import numpy as np
from sklearn.decomposition import PCA
import open3d as o3d


def rotation_matrix_from_axis_angle(axis, angle):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    ux, uy, uz = axis

    return np.array([
        [cos_angle + ux ** 2 * (1 - cos_angle), ux * uy * (1 - cos_angle) - uz * sin_angle,
         ux * uz * (1 - cos_angle) + uy * sin_angle],
        [uy * ux * (1 - cos_angle) + uz * sin_angle, cos_angle + uy ** 2 * (1 - cos_angle),
         uy * uz * (1 - cos_angle) - ux * sin_angle],
        [uz * ux * (1 - cos_angle) - uy * sin_angle, uz * uy * (1 - cos_angle) + ux * sin_angle,
         cos_angle + uz ** 2 * (1 - cos_angle)]
    ])

def refine_flow_direction(pcd):
    """
    Refine the flow direction using Principal Component Analysis (PCA) to find the direction of maximum variation.
    """
    points = np.asarray(pcd.points)

    pca = PCA(n_components=3)
    pca.fit(points)

    flow_direction = pca.components_[0]
    return flow_direction


def pick_direction_interactively(point_cloud):
    print("Press 'Shift + Left Click' to select two points in the Open3D viewer.")

    # Create a copy for interaction
    vis_pcd = point_cloud

    def pick_points(pcd):
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # User picks points
        vis.destroy_window()
        return vis.get_picked_points()

    picked_points_indices = pick_points(vis_pcd)
    if len(picked_points_indices) < 2:
        raise ValueError("You need to pick exactly two points to define a direction.")

    # Get the coordinates of the picked points
    points = np.asarray(point_cloud.points)
    p1, p2 = points[picked_points_indices[0]], points[picked_points_indices[1]]
    direction_vector = p2 - p1
    print(f"Selected direction vector: {direction_vector}")
    return direction_vector, p1, p2