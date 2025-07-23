# main.py
import cv2
from vision import VisionSystem # Import the class from vision.py

def main():
    """
    Example usage of the VisionSystem class.
    This function mimics a main loop (like an RL training loop) where you would
    use the vision system to get data about the environment.
    """
    # --- Configuration ---
    # Choose 'webcam' or 'realsense'
    CAMERA_MODE = 'webcam' 
    
    # Create an instance of the vision system
    # For webcam, you can specify the device ID, e.g., VisionSystem('webcam', 1)
    vision = VisionSystem(camera_type=CAMERA_MODE)
    
    try:
        # Start the camera
        vision.start()

        # --- Main Loop ---
        while True:
            # 1. Get the latest frame from the camera
            frame = vision.get_frame()
            if frame is None:
                print("Failed to grab frame. Exiting.")
                break

            # 2. Use the vision methods to get information
            # Get the angle of the pink "lid"
            angle = vision.get_lid_angle(frame)
            
            # Get detections for red objects (cubes/circles)
            # This method returns an annotated frame and the detection data
            annotated_frame, detections = vision.detect_objects(frame)

            # 3. Print the results (in your RL loop, you'd use this data)
            if angle is not None:
                print(f"Lid Angle: {angle:.2f} degrees")
                # Draw the angle on the frame for visualization
                cv2.putText(annotated_frame, f"Angle: {angle:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                print("Lid not found.")

            if detections:
                print(f"Detected {len(detections)} objects:")
                for det in detections:
                    print(f"  - {det['label']} at {det['centroid']}")
            
            print("-" * 20)


            # 4. Display the output
            cv2.imshow("Vision System Output", annotated_frame)

            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Clean up: stop the camera and close windows
        vision.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
