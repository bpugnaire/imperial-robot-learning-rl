import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Create images directory if it doesn't exist
images_dir = "images"
os.makedirs(images_dir, exist_ok=True)
image_path = os.path.join(images_dir, "test.png")

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Wait for a coherent frame
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
pipeline.stop()

if color_frame:
    color_image = np.asanyarray(color_frame.get_data())
    cv2.imwrite(image_path, color_image)
    print(f"Image saved to {image_path}")
else:
    print("Failed to capture image.")