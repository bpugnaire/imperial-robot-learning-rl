# rl/env.py

import random
import time

class LidOpeningEnv:
    def __init__(self, robot, vision, action_space, angle_threshold=90):
        """
        robot: object with move_joint(action) method
        vision: object with get_lid_angle() method
        action_space: list of possible actions (e.g. [-5, 0, +5] degrees)
        angle_threshold: angle (in degrees) under which the lid is considered open
        """
        self.robot = robot
        self.vision = vision
        self.action_space = action_space
        self.angle_threshold = angle_threshold

        self.state = None

    def reset(self):
        """
        Resets the environment — assumes robot is manually or automatically reset.
        Returns: initial state (angle)
        """
        self.robot.reset_position()
        time.sleep(1)
        frame = self.vision.get_frame()
        angle = self.vision.get_lid_angle(frame)
        self.state = angle
        return angle
    
    def step(self, action_index):
        """
        Applies the action, observes new state, computes reward.
        Returns: next_state, reward, done
        """
        action = self.action_space[action_index]
        success = self.robot.move_joint(action)

        time.sleep(1)  # Wait for motion & image stabilization
        frame = self.vision.get_frame()
        angle = self.vision.get_lid_angle(frame)
        print(f"Lid angle:{angle}")
        reward = self.compute_reward(angle)
        print(f"Reward collected:{reward}")
        done = angle < self.angle_threshold

        self.state = angle
        if not success:
            reward = -100
        return angle, reward, done

    def compute_reward(self, angle):
        # Reward is higher the more the lid is open (lower angle)
        return angle - self.angle
