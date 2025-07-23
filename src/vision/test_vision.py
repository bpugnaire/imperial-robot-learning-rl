# main.py
import cv2
import math

class CubeLidVisionSystem:
    def __init__(self, camera_type='webcam', device_id=0):
        """
        Unified vision system for cube detection and lid angle estimation.

        :param camera_type: 'webcam' or 'realsense'
        :param device_id: for webcam, the device index
        """
        self.camera_type = camera_type
        self.device_id = device_id
        self.cap = None

    def start(self):
        """Initialize the camera capture based on type."""
        if self.camera_type == 'webcam':
            self.cap = cv2.VideoCapture(self.device_id)
        elif self.camera_type == 'realsense':
            # Placeholder: initialize RealSense pipeline
            raise NotImplementedError("RealSense support not implemented yet.")
        else:
            raise ValueError(f"Unknown camera type: {self.camera_type}")

    def stop(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_frame(self):
        """Grab the latest frame from the camera."""
        if not self.cap:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def get_lid_angle(self, frame):
        """Estimate the angle of the pink lid and return annotated frame and angle."""
        annotated = frame.copy()
        # Convert to HSV and threshold pink color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Define pink range (adjust as needed)
        lower_pink = (140, 50, 50)
        upper_pink = (170, 255, 255)
        mask = cv2.inRange(hsv, lower_pink, upper_pink)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return annotated, None

        # Assume largest contour is lid
        lid_contour = max(contours, key=cv2.contourArea)
        if len(lid_contour) < 5:
            # Not enough points to fit ellipse
            return annotated, None

        # Fit ellipse
        ellipse = cv2.fitEllipse(lid_contour)
        (x_center, y_center), (major_axis, minor_axis), angle = ellipse

        # Draw ellipse
        cv2.ellipse(annotated, ellipse, (0, 255, 0), 2)

        # Convert angle to radians
        rad = math.radians(angle)

        # Major axis endpoints
        dx_major = int((major_axis / 2) * math.cos(rad))
        dy_major = int((major_axis / 2) * math.sin(rad))
        pt1_major = (int(x_center - dx_major), int(y_center - dy_major))
        pt2_major = (int(x_center + dx_major), int(y_center + dy_major))
        cv2.line(annotated, pt1_major, pt2_major, (255, 0, 0), 2)

        # Minor axis endpoints (perpendicular)
        rad_perp = rad + math.pi / 2
        dx_minor = int((minor_axis / 2) * math.cos(rad_perp))
        dy_minor = int((minor_axis / 2) * math.sin(rad_perp))
        pt1_minor = (int(x_center - dx_minor), int(y_center - dy_minor))
        pt2_minor = (int(x_center + dx_minor), int(y_center + dy_minor))
        cv2.line(annotated, pt1_minor, pt2_minor, (0, 0, 255), 2)

        # Adjust angle range
        # OpenCV returns angle of the ellipse in degrees
        if angle < -45:
            display_angle = angle + 90
        else:
            display_angle = angle
        # annotated - just for debuging or illustration
        return annotated, display_angle

    def detect_objects(self, frame):
        """Detect red cubes/circles in the frame and annotate."""
        annotated = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = (0, 70, 50)
        upper_red1 = (10, 255, 255)
        lower_red2 = (170, 70, 50)
        upper_red2 = (180, 255, 255)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue

            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            label = 'cube' if len(approx) == 4 else 'circle'

            cv2.drawContours(annotated, [cnt], -1, (0, 255, 0), 2)
            cv2.putText(annotated, label, (cx - 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            detections.append({'label': label, 'centroid': (cx, cy), 'area': area})

        return annotated, detections

    def run(self, mode='both'):
        """
        Run the processing loop.

        :param mode: 'detect', 'angle', or 'both'
        """
        self.start()
        try:
            while True:
                frame = self.get_frame()
                if frame is None:
                    print("Failed to grab frame. Exiting.")
                    break

                annotated = frame.copy()

                if mode in ('angle', 'both'):
                    annotated_angle, angle = self.get_lid_angle(frame)
                    if angle is not None:
                        print(f"Lid Angle: {angle:.2f} degrees")
                        annotated = annotated_angle
                    else:
                        print("Lid not found.")

                if mode in ('detect', 'both'):
                    annotated_detect, detections = self.detect_objects(frame)
                    annotated = annotated_detect
                    if detections:
                        print(f"Detected {len(detections)} objects:")
                        for det in detections:
                            print(f"  - {det['label']} at {det['centroid']}")

                print("-" * 20)

                cv2.imshow("Vision System Output", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    system = CubeLidVisionSystem(camera_type='webcam', device_id=0)
    system.run(mode='angle')


