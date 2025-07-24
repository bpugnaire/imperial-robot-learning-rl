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



class CubeLidVisionSystem:
    def __init__(self, camera_type='webcam', device_id=0):
        """
        Unified vision system for cube detection and lid angle estimation.
        This version uses a single-threaded approach for simple integration.
        """
        self.camera_type = camera_type
        self.device_id = device_id
        self.cap = None

        # --- State Attributes ---
        self.current_lid_angle = None
        self.current_reward = None
        
        # --- Q-Learning Visualization Attributes ---
        self.q_table = None
        self.state_bins = None
        self.actions = None
        self.current_q_state = None
        self.last_action_index = None
        self.flash_counter = 0
        self.flash_duration_frames = 10 # Frames to keep the flash effect

    def set_q_learning_info(self, q_table, state_bins, actions):
        """Set the necessary information for Q-table visualization."""
        self.q_table = q_table
        self.state_bins = state_bins
        self.actions = actions

    def update_q_state(self, state_index, action_index):
        """Update the current state and action to visualize the agent's choice."""
        self.current_q_state = state_index
        self.last_action_index = action_index
        self.flash_counter = self.flash_duration_frames
            
    def set_reward_value(self, reward):
        """Update the reward value."""
        self.current_reward = reward

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
        """Release camera resources and close windows."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Vision system stopped.")

    def _get_frame(self):
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
            return cv2.resize(frame, (1280, 720))
        return None

    def _get_lid_angle(self, frame):
        """Internal method to process lid angle."""
        annotated = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_pink = (140, 50, 50); upper_pink = (170, 255, 255)
        mask = cv2.inRange(hsv, lower_pink, upper_pink)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        angle = None
        if contours:
            lid_contour = max(contours, key=cv2.contourArea)
            if len(lid_contour) >= 5:
                ellipse = cv2.fitEllipse(lid_contour)
                cv2.ellipse(annotated, ellipse, (0, 255, 0), 2)
                angle = ellipse[2]
        
        self.current_lid_angle = angle - 90 if angle else self.current_lid_angle
        return annotated

    def _draw_q_table(self, frame):
        """Internal method to overlay the Q-table visualization."""
        if self.q_table is None:
            return frame

        overlay = frame.copy()
        num_states, num_actions = self.q_table.shape
        start_x, start_y, cell_w, cell_h, header_h = 600, 100, 100, 50, 45
        min_q, max_q = np.min(self.q_table), np.max(self.q_table)

        for i, action in enumerate(self.actions):
            x = start_x + (i + 1) * cell_w
            cv2.putText(overlay, f"A: {action}", (x + 5, start_y + header_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        for s_idx in range(num_states):
            if s_idx < len(self.state_bins):
                state_label = f"S{s_idx} (<{self.state_bins[s_idx]})"
            else:
                # This handles the last state, which is for angles >= the last bin value
                state_label = f"S{s_idx} (>{self.state_bins[-1]})"            
                cv2.putText(overlay, state_label, (start_x, start_y + header_h + s_idx * cell_h + cell_h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            for a_idx in range(num_actions):
                q_value = self.q_table[s_idx, a_idx]
                normalized_q = (q_value - min_q) / (max_q - min_q + 1e-7)
                cell_color = (0, int(200 * normalized_q), 0) if q_value >= 0 else (0, 0, int(200 * abs(normalized_q)))
                if self.flash_counter > 0 and s_idx == self.current_q_state and a_idx == self.last_action_index:
                    cell_color = (0, 255, 255)
                cell_x, cell_y = start_x + (a_idx + 1) * cell_w, start_y + header_h + s_idx * cell_h
                cv2.rectangle(overlay, (cell_x, cell_y), (cell_x + cell_w, cell_y + cell_h), cell_color, -1)
                cv2.rectangle(overlay, (cell_x, cell_y), (cell_x + cell_w, cell_y + cell_h), (255,255,255), 1)
                cv2.putText(overlay, f"{q_value:.2f}", (cell_x + 10, cell_y + cell_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        if self.flash_counter > 0: self.flash_counter -= 1
        return cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    def update_display(self):
        """
        Processes and displays a single frame. Call this in your main loop.
        Returns True if the user has requested to quit, otherwise False.
        """
        frame = self._get_frame()
        if frame is None:
            print("Failed to grab frame.")
            return False # Continue loop, maybe camera is initializing

        annotated = self._get_lid_angle(frame)
        annotated = self._draw_q_table(annotated)
        
        font_color = (0, 255, 0)
        cv2.putText(annotated, f"Reward: {'N/A' if self.current_reward is None else f'{self.current_reward:.2f}'}", (annotated.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, font_color, 2)
        cv2.putText(annotated, f"Lid Angle: {'N/A' if self.current_lid_angle is None else f'{self.current_lid_angle:.2f}'}", (annotated.shape[1] - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, font_color, 2)
        
        cv2.imshow("Vision System Output", annotated)
        
        # Check for 'q' key press to quit
        key = cv2.waitKey(1) & 0xFF
        return key == ord('q')