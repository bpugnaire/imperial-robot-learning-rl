import time
import matplotlib.pyplot as plt  # Add this import

# rl/train.py

from src.rl.lid_opening_env import LidOpeningEnv
from src.rl.agent import QLearningAgent

from src.hardware.python_controller.robot_interface import DummyRobot, Robot
from src.vision.vision import CubeLidVisionSystem
from tqdm import tqdm

port = "/dev/ttyACM0"
bounds = [0.4, 1.1]
robot = Robot(port=port, join_idx=2, bounds=bounds)
# robot = DummyRobot()
vision = CubeLidVisionSystem(camera_type='realsense', device_id=1)

actions = [-15, 0, 15]  # Degrees to rotate the joint
state_bins = list(range(10, 50, 20))  # Bin angle into discrete states

env = LidOpeningEnv(robot, vision, actions, angle_threshold=80)
epsilon = 0.8
agent = QLearningAgent(state_bins=state_bins, num_actions=len(actions), epsilon=epsilon)

steps_per_episode = 4  # Number of steps per episode

cumulative_rewards = []  # Store cumulative rewards per episode

try:
    vision.start()
    vision.set_q_learning_info(agent.q_table, state_bins, actions)
    
    quit_signal_received = False
    for episode in tqdm(range(20)):
        state = agent.discretize_state(env.reset())
        done = False

        transitions = []
        cumulative_reward = 0

        for step in range(steps_per_episode):
            if done:
                break
            action_index = agent.choose_action(state)
            # print(f"Episode {episode}, Step: {step}, State: {state}, Action Index: {action_index}")
            next_angle, reward, done = env.step(action_index)
            vision.set_reward_value(reward)
            vision.update_q_state(state, action_index)
            next_state = agent.discretize_state(next_angle)
            transitions.append((state, action_index, reward, next_state))
            cumulative_reward += reward
            state = next_state

            if vision.update_display():
                # print("Quit signal received from vision system.")
                quit_signal_received = True
                break

        # Update Q-table only at the end of the episode
        if transitions:
            for state, action, _, next_state in transitions:
                agent.update(state, action, cumulative_reward, next_state)

        # print(f"Episode {episode} finished with cumulative reward {cumulative_reward}.")
        cumulative_rewards.append(cumulative_reward)  # Save cumulative reward
        vision.update_display()
        if quit_signal_received:
            break

    # print("Training complete.")

finally:
    vision.stop()
