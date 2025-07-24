import cv2
import numpy as np
import math

from src.vision.vision import CubeLidVisionSystem
# You might need to install pyrealsense2 if you are using a RealSense camera
# import pyrealsense2 as rs




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
