import cv2
import os

def images_to_video(image_folder, output_video_name="output_video.mp4", fps=30):
    """
    Creates a video from all image files in a specified folder.

    Args:
        image_folder (str): The path to the folder containing the images.
        output_video_name (str): The name of the output video file (e.g., "my_video.mp4").
        fps (int): Frames per second for the output video.
    """
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))]
    images.sort()  # Sort the images to ensure correct frame order

    if not images:
        print(f"No images found in the folder: {image_folder}")
        return

    # Read the first image to get dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Error: Could not read the first image: {first_image_path}")
        return

    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    # For MP4, 'mp4v' or 'H264' are common codecs.
    # Check your OpenCV build for supported codecs if you encounter issues.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_video_name}. Check codecs or file path.")
        return

    print(f"Starting video creation from {len(images)} images...")
    for i, image in enumerate(images):
        image_path = os.path.join(image_folder, image)
        img = cv2.imread(image_path)

        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue

        # Resize image to match the video dimensions if necessary
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height))

        out.write(img)
        print(f"Processed image {i+1}/{len(images)}: {image}", end='\r') # Progress indicator

    out.release()
    cv2.destroyAllWindows()
    print(f"\nVideo '{output_video_name}' created successfully in the current directory.")

if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Replace 'path/to/your/images' with the actual path to your image folder.
    image_folder_path = "debugging"
    output_video_filename = "eps03.mp4"  # Name of the output video file
    frames_per_second = 10 # Adjust as needed (e.g., 24, 30, 60)

    # --- Run the script ---
    if os.path.isdir(image_folder_path):
        images_to_video(image_folder_path, output_video_filename, frames_per_second)
    else:
        print(f"Error: The specified image folder does not exist: {image_folder_path}")
        print("Please update 'image_folder_path' in the script to the correct path.")

    # Example of how to create a dummy folder with images for testing:
    # import numpy as np
    # if not os.path.exists("test_images"):
    #     os.makedirs("test_images")
    # for i in range(10):
    #     dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    #     cv2.putText(dummy_image, f"Frame {i+1}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    #     cv2.imwrite(f"test_images/frame_{i:03d}.png", dummy_image)
    # print("Created 'test_images' folder with 10 dummy images for testing.")
    # print("You can now set 'image_folder_path = \"test_images\"' to test the script.")