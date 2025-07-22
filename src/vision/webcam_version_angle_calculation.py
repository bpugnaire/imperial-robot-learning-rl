import cv2
import numpy as np
import math
import pyrealsense2 as rs
import matplotlib.pyplot as plt

# In Spyder, set IMAGE_PATH to a valid image file path to test on a still image.
# Otherwise, set to None to use webcam or RealSense based on MODE.
IMAGE_PATH = r"C:\Users\khelvig\Downloads\test.png"  # or None
MODE = 'webcam'  # choose 'image', 'webcam', or 'realsense'
PARALLEL_ANGLE_THRESH = 10  # degrees to filter near-horizontal segments

def detect_and_draw_segments(frame, segments):
    """
    Draw detected line segments in red on the frame.
    """
    for x1, y1, x2, y2 in segments:
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    return frame

def estimate_line_angle_horizontal(frame, min_length=120):
    """
    Estimate line orientation relative to horizontal in degrees for blue regions.
    Filter out segments whose angle to horizontal is within PARALLEL_ANGLE_THRESH.
    Returns (angle_degrees or None, list_of_segments).
    """
    # 1. Mask blue regions in HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    filtered = cv2.bitwise_and(frame, frame, mask=mask)
    # plt.imshow(filtered); plt.title("Filtered (blue mask)"); plt.figure()

    # 2. Convert to grayscale, resize, and blur
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (640, 480))
    gray = cv2.blur(gray, (5, 5))

    # 3. Enhance contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(3, 3))
    gray = clahe.apply(gray)
    # plt.imshow(gray); plt.title("CLAHE result"); plt.figure()

    # 4. Detect line segments with LSD
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = lsd.detect(gray)
    lines = detected[0] if detected is not None and detected[0] is not None else []

    points = []
    segments = []
    # 5. Filter by length and near-horizontal
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length < min_length:
            continue
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        ang_abs = abs(ang)
        if ang_abs < PARALLEL_ANGLE_THRESH or abs(ang_abs - 180) < PARALLEL_ANGLE_THRESH:
            continue
        segments.append((x1, y1, x2, y2))
        points.extend([(x1, y1), (x2, y2)])

    if len(points) < 2:
        return None, segments

    # 6. Fit a single line to remaining points
    vx, vy, _, _ = cv2.fitLine(
        np.array(points), cv2.DIST_L2, 0, 0.01, 0.01
    ).flatten()

    # 7. Compute angle relative to horizontal
    angle_rad = math.atan2(vy, vx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg, segments

def process_image(path):
    """
    Process a single image file and display the annotated result.
    """
    frame = cv2.imread(path)
    if frame is None:
        print(f"Error: cannot read image from {path}")
        return
    frame = cv2.resize(frame, (640, 480))

    angle, segments = estimate_line_angle_horizontal(frame)
    annotated = detect_and_draw_segments(frame.copy(), segments)
    if angle is not None:
        cv2.putText(
            annotated,
            f"Angle: {angle:.1f}°",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    plt.imshow(annotated)
    plt.title("Line Orientation - Image")
    plt.show()

def process_webcam():
    """
    Process live video from default webcam. Press 'q' or Esc to exit.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        angle, segments = estimate_line_angle_horizontal(frame)
        annotated = detect_and_draw_segments(frame.copy(), segments)
        if angle is not None:
            cv2.putText(
                annotated,
                f"Angle: {angle:.1f}°",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )
        cv2.imshow("Line Orientation - Webcam", annotated)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def process_realsense():
    """
    Process live video from Intel RealSense (RGB only). Press 'q' or Esc to exit.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            frame = cv2.resize(frame, (640, 480))
            angle, segments = estimate_line_angle_horizontal(frame)
            annotated = detect_and_draw_segments(frame.copy(), segments)
            if angle is not None:
                cv2.putText(
                    annotated,
                    f"Angle: {angle:.1f}°",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )
            cv2.imshow("Line Orientation - RealSense", annotated)
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if MODE == 'image' and IMAGE_PATH:
        process_image(IMAGE_PATH)
    elif MODE == 'webcam':
        process_webcam()
    elif MODE == 'realsense':
        process_realsense()
    else:
        print("Invalid mode or missing IMAGE_PATH for 'image'.")
