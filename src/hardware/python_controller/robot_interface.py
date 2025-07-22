# robot_interface.py
import time
from adeept_comms import RobotConnection

class Robot:
    """
    A simplified, high-level interface for controlling a single joint of the 
    Adeept Robotic Arm, designed for reinforcement learning applications.
    This class now correctly uses the RobotConnection class for communication.
    """

    def __init__(self, port='COM4', joint_id=1):
        """
        Initializes the connection to the robotic arm.
        On macOS or Linux: Open a terminal and run ls /dev/tty.* or ls /dev/ttyUSB*. 
        The board will appear as something like /dev/tty.usbmodem14101 or /dev/ttyUSB0.

        Args:
            port (str): The serial port the robot is connected to (e.g., 'COM4').
            joint_id (int): The specific servo joint this instance will control (1-6).
        """
        self.joint_id = joint_id
        self.home_angle = 90
        self.current_angle = self.home_angle
        
        # --- Connection Initialization ---
        # Create an instance of the low-level communication class.
        self.connection = RobotConnection()

        try:
            # Use the methods from the RobotConnection instance to connect.
            if not self.connection.connect(port):
                # If connect returns False, raise an exception.
                raise ConnectionError(f"Failed to establish serial connection on port {port}.")
            
            # Perform the initial handshake to ensure the robot is ready.
            self.connection.wait_for_setup_confirmation()
            
            print(f"Robot connection established. Controlling Joint ID: {self.joint_id}")
            
            # Move the joint to its starting home position.
            self.reset_position()

        except Exception as e:
            print(f"Failed during robot initialization: {e}")
            # Ensure disconnection on failure.
            self.disconnect()
            raise

    def _move_servo_to(self, angle, move_time=250):
        """
        Private helper method to move the specified joint to an absolute angle.
        
        Args:
            angle (int): The absolute angle to move the servo to.
            move_time (int): The time in milliseconds for the movement.
        """
        # Clamp the angle to the valid hardware range (0-180 degrees).
        angle = int(max(0, min(180, angle)))
        
        # Use the send_command method from our connection object.
        self.connection.send_command('servo', self.joint_id, angle, move_time)
        
        # Update the stored state of the joint's angle.
        self.current_angle = angle
        print(f"Joint {self.joint_id} moved to {self.current_angle}°.")

    def move_joint(self, action):
        """
        Applies a relative angle change (an action) to the joint.

        Args:
            action (int): The relative angle to add/subtract (e.g., -5, 0, 5).

        Returns:
            int: The new absolute angle of the joint after the action.
        """
        new_angle = self.current_angle + action
        self._move_servo_to(new_angle)
        return self.current_angle

    def reset_position(self):
        """Moves the joint to its predefined home position."""
        print(f"Resetting Joint {self.joint_id} to home position ({self.home_angle}°).")
        self._move_servo_to(self.home_angle)

    def disconnect(self):
        """Closes the serial connection to the robot."""
        print("Disconnecting from robot.")
        if self.connection:
            self.connection.disconnect()

# Example of how to use the corrected Robot class:
if __name__ == '__main__':
    # This block will only run when robot_interface.py is executed directly.
    # Replace 'COM4' with your robot's actual serial port.
    COM_PORT = 'COM4'

    robot_joint = None
    try:
        # Initialize the robot to control a specific joint.
        robot_joint = Robot(port=COM_PORT, joint_id=1)
        time.sleep(1)

        # Define a sequence of actions.
        actions = [-5, -5, -10, -10, 20, 10]
        
        print("\n--- Starting relative movement sequence ---")
        for act in actions:
            print(f"\nApplying action: {act}°")
            robot_joint.move_joint(act)
            time.sleep(1)
        
        print("\n--- Sequence complete. Resetting to home. ---")
        robot_joint.reset_position()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Ensure the connection is closed cleanly.
        if robot_joint:
            robot_joint.disconnect()
