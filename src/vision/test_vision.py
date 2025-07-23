import cv2
import numpy as np
import math

# You might need to install pyrealsense2 if you are using a RealSense camera
# import pyrealsense2 as rs

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
        """Initialize the camera."""
        if self.camera_type == 'webcam':
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                raise IOError(f"Cannot open webcam with device_id {self.device_id}")
        # ... (RealSense initialization would go here)
        else:
            raise ValueError(f"Unknown camera type: {self.camera_type}")
        print("Vision system started.")

    def stop(self):
        """Release camera resources and close windows."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Vision system stopped.")

    def _get_frame(self):
        """Internal method to grab a frame from the camera."""
        if not self.cap or not self.cap.isOpened(): return None
        ret, frame = self.cap.read()
        if ret: frame = cv2.flip(frame, 1)
        return frame if ret else None

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
        
        self.current_lid_angle = angle
        return annotated

    def _draw_q_table(self, frame):
        """Internal method to overlay the Q-table visualization."""
        if self.q_table is None:
            return frame

        overlay = frame.copy()
        num_states, num_actions = self.q_table.shape
        start_x, start_y, cell_w, cell_h, header_h = 20, 70, 100, 50, 45
        min_q, max_q = np.min(self.q_table), np.max(self.q_table)

        for i, action in enumerate(self.actions):
            x = start_x + (i + 1) * cell_w
            cv2.putText(overlay, f"A: {action}", (x + 5, start_y + header_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        for s_idx in range(num_states):
            state_label = f"S{s_idx} (<{self.state_bins[s_idx]})"
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


if __name__ == '__main__':
    actions = [-10, 0, 10]
    state_bins = [30, 50, 70, 90, 181]
    q_table = np.zeros((len(state_bins), len(actions)))

    vision_system = CubeLidVisionSystem(camera_type='webcam', device_id=0)
    
    try:
        vision_system.start()
        vision_system.set_q_learning_info(q_table, state_bins, actions)

        print("--- Starting Mock Training Loop ---")
        # This loop simulates your main training script
        for episode in range(500):
            # --- Your RL Logic Goes Here ---
            state_idx = np.random.randint(0, len(state_bins))
            action_idx = np.random.randint(0, len(actions))
            print(f"Episode {episode}: Simulating action {actions[action_idx]} in state {state_idx}")
            
            # Update the vision system with the latest info
            vision_system.update_q_state(state_idx, action_idx)
            reward = np.random.random() - 0.5
            q_table[state_idx, action_idx] += reward
            vision_system.set_reward_value(reward)
            # --- End of RL Logic ---

            # --- A SINGLE LINE handles all vision processing and display ---
            if vision_system.update_display():
                print("Quit signal received.")
                break

        print("--- Mock Training Loop Finished ---")

    finally:
        vision_system.stop()
