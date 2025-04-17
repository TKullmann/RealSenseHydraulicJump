import pyrealsense2 as rs
import numpy as np
import open3d as o3d


def process_bag_file(bag_file, output_ply=None):
    # Initialize pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Configure to read from .bag file
    config.enable_device_from_file(bag_file)
    pipeline.start(config)

    for _ in range(5):  # Skip the first few frames for better alignment
        pipeline.wait_for_frames()

    frameset = pipeline.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    pipeline.stop()
    print("Frames captured")

    # Convert frames to numpy arrays
    color = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data())

    # Align depth to color
    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)
    aligned_depth_frame = frameset.get_depth_frame()
    depth = np.asanyarray(aligned_depth_frame.get_data())

    # Intrinsics
    depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    fx, fy = depth_intrinsics.fx, depth_intrinsics.fy
    cx, cy = depth_intrinsics.ppx, depth_intrinsics.ppy

    # Generate point cloud
    points = []
    colors = []
    height, width = depth.shape
    for v in range(height):
        for u in range(width):
            z = depth[v, u] * 0.001  # Depth in meters
            if z == 0:  # Skip invalid points
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])
            colors.append(color[v, u] / 255.0)  # Normalize RGB to [0, 1]

    # Convert to Open3D PointCloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))

    # Save the point cloud to a .ply file
    if output_ply:
        o3d.io.write_point_cloud(output_ply, point_cloud)
        print(f"Point cloud saved to {output_ply}")

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])


# Example usage
process_bag_file("../dataPraktikumsrinneLong/angle_full_cycle.bag", output_ply="output3.ply")
