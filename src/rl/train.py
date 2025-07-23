# rl/train.py

from rl.lid_opening_env import LidOpeningEnv
from rl.agent import QLearningAgent

# Dummy robot and vision classes for now
from hardware.python_controller.robot_interface import DummyRobot, Robot
from vision.vision import VisionSystem


port = "/dev/ttyACM0"

robot = Robot(port=port, join_idx=3)
vision = VisionSystem()
vision.start()
actions = [-5, 0, 5]  # Degrees to rotate the joint
state_bins = [10, 30, 60, 90]  # Bin angle into discrete states

env = LidOpeningEnv(robot, vision, actions)
agent = QLearningAgent(state_bins=state_bins, num_actions=len(actions))

for episode in range(100):
    state = agent.discretize_state(env.reset())
    done = False

    while not done:
        action = agent.choose_action(state)
        next_angle, reward, done = env.step(action)
        next_state = agent.discretize_state(next_angle)

        agent.update(state, action, reward, next_state)
        state = next_state

    print(f"Episode {episode} complete")
