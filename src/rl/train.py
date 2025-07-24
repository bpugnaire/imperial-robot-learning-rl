import time
# rl/train.py

from src.rl.lid_opening_env import LidOpeningEnv
from src.rl.agent import QLearningAgent

from src.hardware.python_controller.robot_interface import DummyRobot, Robot
from src.vision.vision import CubeLidVisionSystem


port = "/dev/ttyACM0"
bounds = [0.4, 1.1]
# robot = Robot(port=port, join_idx=3, bounds=bounds)
robot = DummyRobot()
vision = CubeLidVisionSystem()

actions = [-10, 0, 10]  # Degrees to rotate the joint
state_bins = [30, 50, 70, 100]  # Bin angle into discrete states

env = LidOpeningEnv(robot, vision, actions, angle_threshold=80)
agent = QLearningAgent(state_bins=state_bins, num_actions=len(actions))

try:
    vision.start()
    # Pass the Q-table and other info to the vision system for visualization
    vision.set_q_learning_info(agent.q_table, state_bins, actions)
    
    quit_signal_received = False
    for episode in range(100):
        state = agent.discretize_state(env.reset())
        print("this is the state from dicretize", state)

        done = False

        while not done:
            # Wait for user to press Enter before proceeding to the next action
            input("Press Enter to perform the next action...")

            # The agent gives us the index of the action to take
            action_index = agent.choose_action(state)
            next_angle, reward, done = env.step(action_index)
            
            # Update the vision system with the latest information
            vision.set_reward_value(reward)
            vision.update_q_state(state, action_index)

            if done:
                print(f"Episode {episode} finished with reward {reward}.")
                # Update display one last time before breaking
                vision.update_display()
                time.sleep(1) # Pause for a moment on the final state
                break

            next_state = agent.discretize_state(next_angle)

            # The agent's update function needs the action index
            agent.update(state, action_index, reward, next_state)
            state = next_state

            if vision.update_display():
                print("Quit signal received from vision system.")
                quit_signal_received = True
                break
        if quit_signal_received:
            break

    print("Training complete.")

finally:
    vision.stop()

