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
    #p1 = np.asarray((-0.29, -0.23, 0.61))
    #p2 = np.asarray((-0.43, 0.36, 0.82))
    direction_vector = p2 - p1
    print(f"Selected direction vector: {direction_vector}")
    return direction_vector, p1, p2

def fit_plane_to_point_cloud(pcd):
    # Fit a plane to the point cloud using RANSAC
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                             ransac_n=3,
                                             num_iterations=1000)
    normal_vector = plane_model[:3]  # The normal vector of the plane
    return normal_vector

def define_plane_from_two_points_and_normal(p1, p2, normal):
    # Create a plane that passes through p1 and p2 and is orthogonal to `normal`
    p1 = np.array(p1)
    p2 = np.array(p2)
    normal = np.array(normal)

    # The direction vector between p1 and p2
    direction_vector = p2 - p1
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    # Compute a second orthogonal vector to form the plane
    orthogonal_vector = np.cross(direction_vector, normal)
    orthogonal_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)

    return p1, direction_vector, orthogonal_vector

def visualize_plane(pcd, p1, p2, normal):
    import open3d as o3d

    # Define the size of the plane
    grid_size = 1.0
    num_points = 20

    # Convert points and normal to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)
    normal = np.array(normal)

    # Calculate the direction vector between p1 and p2
    direction_vector = p2 - p1
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    # Generate a grid of points on the orthogonal plane
    x = np.linspace(-grid_size, grid_size, num_points)
    y = np.linspace(-grid_size, grid_size, num_points)
    xx, yy = np.meshgrid(x, y)
    plane_points = p1 + xx[:, :, None] * direction_vector + yy[:, :, None] * normal
    plane_points = plane_points.reshape(-1, 3)

    # Create triangles for the mesh
    triangles = []
    for i in range(num_points - 1):
        for j in range(num_points - 1):
            idx = i * num_points + j
            triangles.append([idx, idx + 1, idx + num_points])
            triangles.append([idx + 1, idx + num_points, idx + num_points + 1])

    # Create the Open3D TriangleMesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(plane_points)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([1.0, 0.0, 0.0])  # Red color for the orthogonal plane

    # Visualize the point cloud and the orthogonal plane
    #o3d.visualization.draw_geometries([pcd, mesh])


def align_to_xy_plane(points, normal):
    """
    Aligns the plane with the XY plane by calculating the rotation matrix.
    """
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)

    # Target normal is along the Z-axis
    target_normal = np.array([0, 0, 1])

    # Compute rotation axis (cross product) and angle
    rotation_axis = np.cross(normal, target_normal)
    if np.linalg.norm(rotation_axis) < 1e-6:  # Already aligned
        return points  # No rotation needed
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    angle = np.arccos(np.dot(normal, target_normal))

    # Rodrigues' rotation formula
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    # Apply rotation to all points
    aligned_points = points @ R.T
    return aligned_points

