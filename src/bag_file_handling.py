import re

import pyrealsense2 as rs
import numpy as np
import cv2

def read_bag_file(bag_file, start_frame, frame_count):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file)  # Load the .bag file
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)

    pipeline.start(config)
    align = rs.align(rs.stream.color)  # Align depth to color frame

    # Skip frames up to the starting frame number
    for _ in range(start_frame):
        pipeline.wait_for_frames()

    depth_accumulator = None
    color_accumulator = None
    valid_frame_count = 0

    for i in range(frame_count):
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue  # Skip frames if either depth or color is missing

        # Convert to NumPy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        match = re.search(r'(\d{8}_\d{6})', bag_file)
        number = match.group()

        if i == 0:
            cv2.imwrite(f'../outputs/frame_{number}.png', cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))

        # Initialize accumulators if not already done
        if depth_accumulator is None:
            depth_accumulator = np.zeros_like(depth_image, dtype=np.float64)
            color_accumulator = np.zeros_like(color_image, dtype=np.float64)

        # Accumulate the frames
        depth_accumulator += depth_image
        color_accumulator += color_image
        valid_frame_count += 1

    if valid_frame_count == 0:
        raise RuntimeError("No valid frames found in the .bag file")

    # Compute the averages
    depth_average = (depth_accumulator / valid_frame_count).astype(np.float32)
    color_average = (color_accumulator / valid_frame_count).astype(np.uint8)

    # Save outputs

    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_average, alpha=0.03),
        cv2.COLORMAP_HOT
    )
    cv2.imwrite('../outputs/averaged_depth_colormap.png', depth_colormap)

    depth_scale = depth_frame.get_units()  # Get depth scale in meters
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

    pipeline.stop()
    return depth_average, color_average, depth_scale, intrinsics
