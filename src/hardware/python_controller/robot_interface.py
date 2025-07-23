# robot.py
import src.hardware.python_controller.bus as bus
from reset_positon import reset_robot_position
from src.hardware.python_controller.bus import FeetechBus
from src.hardware.python_controller.move_join import move_join
class DummyRobot:
    """
    A dummy robot class that simulates robot actions for testing purposes.
    It conforms to the interface required by the LidOpeningEnv.
    """
    def __init__(self, initial_joint_angle=0):
        """
        Initializes the robot's state.

        Args:
            initial_joint_angle (int): The starting angle of the robot's joint.
        """
        self.initial_joint_angle = initial_joint_angle
        self.current_joint_angle = self.initial_joint_angle
        print("DummyRobot initialized.")

    def reset_position(self):
        """
        Resets the robot's joint to the initial position.
        """
        self.current_joint_angle = self.initial_joint_angle
        print(f"Robot position reset to {self.current_joint_angle} degrees.")

    def move_joint(self, action):
        """
        Simulates moving the robot's joint by a certain amount.

        Args:
            action (float): The amount to change the joint angle (in degrees).
        """
        self.current_joint_angle += action
        print(f"Robot moved by {action} degrees. New angle: {self.current_joint_angle} degrees.")


class Robot():
    """
    A dummy robot class that simulates robot actions for testing purposes.
    It conforms to the interface required by the LidOpeningEnv.
    """
    def __init__(self, port, join_idx):
        """
        Initializes the robot's state.

        Args:
            initial_joint_angle (int): The starting angle of the robot's joint.
        """

        self.join_idx = join_idx
        self.bus = FeetechBus(port, [1, 2, 3, 4, 5, 6], calib_file="so101_calibration.json")
        print("Robot initialized.")

    def reset_position(self):
        """
        Resets the robot's joint to the initial position.
        """
        reset_robot_position()
    
    def _get_current_join_positions():
        return bus.get_qpos()

    def move_joint(self, action):
        """
        Simulates moving the robot's joint by a certain amount.

        Args:
            action (float): The amount to change the joint angle (in degrees).
        """
        
        current_pos = self._get_current_join_positions()
        target_pos = current_pos.copy()
        target_pos[self.join_idx] += action
        move_join(current_pos, target_pos, duration=2.0, steps=100)