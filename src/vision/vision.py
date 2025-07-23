# vision.py
import cv2
import numpy as np
import math
from abc import ABC, abstractmethod

# Try to import the RealSense library, but don't fail if it's not installed,
# so the code can still run in webcam mode.
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

class Vision(ABC):
    """
    An abstract base class for a vision system. It defines the standard
    interface for initializing, capturing frames, and performing vision tasks.
    """

    @abstractmethod
    def start(self):
        """Initializes and starts the camera stream."""
        pass

    @abstractmethod
    def stop(self):
        """Stops the camera stream and releases resources."""
        pass

    @abstractmethod
    def get_frame(self):
        """Captures and returns a single frame from the camera."""
        pass

    @abstractmethod
    def get_lid_angle(self, frame):
        """
        Calculates the angle of the lid (pink blob) in the given frame.
        
        Args:
            frame: The input image frame.
            
        Returns:
            The calculated angle in degrees, or None if not found.
        """
        pass

    @abstractmethod
    def detect_objects(self, frame):
        """
        Detects objects (red cubes and circles) in the given frame.

        Args:
            frame: The input image frame.

        Returns:
            A tuple containing the annotated frame and a list of detected objects.
        """
        pass


class VisionSystem(Vision):
    """
    A concrete implementation of the Vision class that uses the algorithms
    from the provided scripts. It can use either a standard webcam or an
    Intel RealSense camera.
    """

    def __init__(self, camera_type='webcam', device_id=0):
        """
        Initializes the Vision System.

        Args:
            camera_type (str): The type of camera to use ('webcam' or 'realsense').
            device_id (int): The device ID for the webcam.
        """
        if camera_type not in ['webcam', 'realsense']:
            raise ValueError("camera_type must be 'webcam' or 'realsense'")

        if camera_type == 'realsense' and not REALSENSE_AVAILABLE:
            raise RuntimeError("RealSense camera selected but pyrealsense2 library is not installed.")
            
        self.camera_type = camera_type
        self.device_id = device_id
        self.pipeline = None
        self.cap = None

        # --- Parameters from provided scripts ---
        # For pink blob detection
        self.MIN_BLOB_AREA = 300
        self.LOWER_PINK1 = np.array([145, 50, 50])
        self.UPPER_PINK1 = np.array([180, 255, 255])
        self.LOWER_PINK2 = np.array([0, 50, 50])
        self.UPPER_PINK2 = np.array([10, 255, 255])

        # For red object detection
        self.CIRCULARITY_THRESHOLD = 0.75
        self.BORDER_MARGIN = 20

    def start(self):
        """Initializes and starts the camera stream."""
        if self.camera_type == 'realsense':
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            self.pipeline.start(config)
        else: # webcam
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                raise IOError(f"Cannot open webcam with device_id {self.device_id}")
        print(f"{self.camera_type.capitalize()} camera started.")

    def stop(self):
        """Stops the camera stream and releases resources."""
        if self.camera_type == 'realsense' and self.pipeline:
            self.pipeline.stop()
        elif self.cap:
            self.cap.release()
        print(f"{self.camera_type.capitalize()} camera stopped.")

    def get_frame(self):
        """
        Captures and returns a single frame from the camera.
        Returns None if the frame cannot be captured.
        """
        frame = None
        if self.camera_type == 'realsense':
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                frame = np.asanyarray(color_frame.get_data())
        else: # webcam
            ret, frame = self.cap.read()
            if not ret:
                return None
        
        # Always resize to a standard dimension for consistency
        if frame is not None:
            return cv2.resize(frame, (640, 480))
        return None

    def _enhance_frame(self, frame):
        """Enhances a frame for better blob detection."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.equalizeHist(v)
        hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def get_lid_angle(self, frame):
        """
        Calculates the angle of the lid (pink blob) in the given frame.
        This method is based on detector_pink_blob_angleEstimation_camerastream.py
        """
        # 1. Enhance and create a mask for pink color
        proc_frame = self._enhance_frame(frame)
        hsv = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.LOWER_PINK1, self.UPPER_PINK1)
        mask2 = cv2.inRange(hsv, self.LOWER_PINK2, self.UPPER_PINK2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 2. Find the largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < self.MIN_BLOB_AREA or len(largest_contour) < 5:
            return None

        # 3. Fit an ellipse and calculate the angle
        ellipse = cv2.fitEllipse(largest_contour)
        # ellipse: ((cx, cy), (axis1, axis2), raw_angle)
        (_, (axis1, axis2), raw_angle) = ellipse

        # if axis1 >= axis2:
        #     # The raw_angle corresponds to the major axis
        #     angle_rad = math.radians(raw_angle)
        # else:
        #     # The raw_angle corresponds to the minor axis, add 90 degrees
        #     angle_rad = math.radians(raw_angle + 90)
        angle_rad = math.radians(raw_angle)

        # # Normalize angle to be acute (0-90 degrees) relative to horizontal
        angle_deg = math.degrees(angle_rad)
        # if angle_deg > 90:
        #     angle_deg = 180 - angle_deg

        return angle_deg

    def detect_objects(self, frame):
        """
        Detects red cubes and circles in the frame.
        Based on Cube_circle_detector_withvideoinput_realsense.py
        """
        h, w = frame.shape[:2]
        output_frame = frame.copy()
        results = []

        # 1. Convert to HSV and mask red regions
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([179, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # 2. Noise reduction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 3. Find contours and classify shapes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 220:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            if (x <= self.BORDER_MARGIN or y <= self.BORDER_MARGIN or
                x + cw >= w - self.BORDER_MARGIN or y + ch >= h - self.BORDER_MARGIN):
                continue
            
            M = cv2.moments(cnt)
            if M['m00'] == 0: continue
            cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            label = 'Circle' if circularity >= self.CIRCULARITY_THRESHOLD else 'Cube'
            
            results.append({'label': label, 'centroid': (cx, cy), 'area': area})
            
            # Draw on the output frame for visualization
            cv2.drawContours(output_frame, [cnt], -1, (0, 255, 255), 2)
            cv2.putText(output_frame, label, (cx - 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return output_frame, results

