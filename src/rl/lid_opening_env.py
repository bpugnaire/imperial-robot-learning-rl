# rl/env.py

import random
import time

class LidOpeningEnv:
    def __init__(self, robot, vision, action_space, angle_threshold=80):
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

        self.state = 0

    def reset(self):
        """
        Resets the environment — assumes robot is manually or automatically reset.
        Returns: initial state (angle)
        """
        self.robot.reset_position()
        time.sleep(1)
        frame = self.vision._get_frame()
        
        self.vision._get_lid_angle(frame)
        angle = self.vision.current_lid_angle
        self.state = angle
        return angle
    
    def step(self, action_index):
        """
        Applies the action, observes new state, computes reward.
        Returns: next_state, reward, done
        """
        action = self.action_space[action_index]
        allowed_move = self.robot.move_joint(action)
        # print(f"Action taken: {action} degrees, Allowed: {allowed_move}")
        time.sleep(.1)  # Wait for motion & image stabilization
        frame = self.vision._get_frame()
        self.vision._get_lid_angle(frame)
        angle = self.vision.current_lid_angle

        # print(f"Lid angle:{angle}")
        reward = angle  # Just return the angle for accumulation
        # print(f"Reward collected:{reward}")
        done = (angle >= self.angle_threshold)

        self.state = angle
        # if not allowed_move:
        #     reward = -100
        return angle, reward, done
