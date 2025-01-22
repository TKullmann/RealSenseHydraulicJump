import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
import open3d as o3d


def parse_bag_file():
    # Specify bag file path and output directory
    bag_file = "../dataPraktikumsrinne/Praktikumsrinne2.bag"
    output_dir = f"{bag_file}_output"
    os.makedirs(output_dir, exist_ok=True)
    # Initialize pipeline and configuration
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file, repeat_playback=False)
    # Start the pipeline
    pipeline.start(config)
    device = pipeline.get_active_profile().get_device()
    # Playback control to access playback options
    playback = device.as_playback()
    playback.set_real_time(False)  # Disable real-time playback for frame-by-frame access
    # Metadata storage
    metadata = []
    try:
        while True:
            # Wait for a new frame
            frames = pipeline.wait_for_frames()

            # Extract depth and color frames
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Save depth frame as a numpy array
            depth_data = np.asanyarray(depth_frame.get_data())
            depth_file = os.path.join(output_dir, f"depth_{depth_frame.get_frame_number()}.npy")
            np.save(depth_file, depth_data)
            depth_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
            depth_inverted = 255 - depth_normalized  # Invert the scale
            depth_image = np.uint8(depth_inverted)
            cv2.imwrite(os.path.join(output_dir, f"depth_{depth_frame.get_frame_number()}.png"), depth_image)

            # Save color frame as an image
            color_data = np.asanyarray(color_frame.get_data())
            color_file = os.path.join(output_dir, f"color_{color_frame.get_frame_number()}.png")
            cv2.imwrite(color_file, cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR))

            # Collect metadata
            frame_metadata = {
                "frame_number": depth_frame.get_frame_number(),
                "timestamp": depth_frame.get_timestamp(),
                "depth_scale": depth_frame.get_units(),
            }
            metadata.append(frame_metadata)

    except RuntimeError:  # Occurs when the bag file ends
        print("Finished reading all frames.")

    finally:
        pipeline.stop()
    # Save metadata to JSON
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Data exported to {output_dir}")


parse_bag_file()

def extract_pcd():
    vertices = []
    depths = np.load("../output/depth_12694.npy")
    for x in range(depths.shape[0]):
        for y in range(depths.shape[1]):
            vertices.append((float(x), float(y), depths[x][y]))
    pcd = o3d.geometry.PointCloud()
    point_cloud = np.asarray(np.array(vertices))
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.estimate_normals()
    pcd = pcd.normalize_normals()
    o3d.io.write_point_cloud("../output.ply", pcd)


#extract_pcd()
