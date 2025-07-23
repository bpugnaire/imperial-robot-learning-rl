# robot.py

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

