# adeept_comms.py
# This is the refactored low-level communication library for the Adeept Robotic Arm.
# It handles the serial connection and the specific command protocol required by the firmware.

import serial
import time

class RobotConnection:
    """
    Manages the serial connection and communication protocol for the Adeept robot.
    This class encapsulates the low-level details of sending commands and receiving
    responses from the robot's controller board.
    """

    def __init__(self):
        """Initializes the RobotConnection instance."""
        self.serial_connection = None

    def connect(self, port, baudrate=115200, timeout=3, write_timeout=10):
        """
        Establishes the serial connection to the robot.

        Args:
            port (str): The serial port name (e.g., 'COM4' or '/dev/ttyUSB0').
            baudrate (int): The communication speed.
            timeout (int): Read timeout in seconds.
            write_timeout (int): Write timeout in seconds.
        
        Returns:
            bool: True if connection was successful, False otherwise.
        """
        try:
            self.serial_connection = serial.Serial(
                port, baudrate, timeout=timeout, writeTimeout=write_timeout
            )
            print(f"Serial connection opened on {port}.")
            return True
        except serial.SerialException as e:
            print(f"Error: Failed to connect on port {port}. {e}")
            self.serial_connection = None
            return False

    def disconnect(self):
        """Closes the serial connection if it is open."""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            print("Serial connection closed.")

    def _format_command(self, *args):
        """
        Formats a command and its arguments into the specific string protocol
        that the robot's firmware expects.

        Example: _format_command('servo', 1, 90) -> "{'start':['servo','1','90']}\n"
        
        Args:
            *args: A variable number of arguments for the command.

        Returns:
            str: The formatted command string.
        """
        # Convert all arguments to strings
        str_args = [str(arg) for arg in args]
        # Join them with commas
        command_payload = ",".join(str_args)
        # Construct the final command string
        return f"{{'start':[{command_payload}]}}\n"

    def send_command(self, *args):
        """
        Sends a command to the robot without waiting for a response.

        Args:
            *args: The command and its parameters.
        """
        if not self.serial_connection:
            print("Error: Not connected. Cannot send command.")
            return

        command_string = self._format_command(*args)
        self.serial_connection.write(command_string.encode("gbk"))

    def send_command_and_wait_for_response(self, *args):
        """
        Sends a command and waits indefinitely for a line of response from the robot.

        Args:
            *args: The command and its parameters.

        Returns:
            str: The response from the robot as a string, or None if not connected.
        """
        if not self.serial_connection:
            print("Error: Not connected. Cannot send command.")
            return None
            
        command_string = self._format_command(*args)
        
        while True:
            self.serial_connection.write(command_string.encode("gbk"))
            response = self.serial_connection.readline()
            if response:
                # Decode from bytes to string and strip whitespace
                return response.decode('gbk').strip()

    def wait_for_setup_confirmation(self):
        """
        Continuously sends a 'setup' command until it receives any response,
        confirming the robot is ready. This is used for the initial handshake.
        """
        print("Waiting for robot to confirm setup...")
        while True:
            command_string = "{'start':['setup']}\n"
            self.serial_connection.write(command_string.encode("gbk"))
            line = self.serial_connection.readline()
            if line:
                print("Robot setup confirmed.")
                break
            time.sleep(0.1)

# Example usage (for testing purposes)
if __name__ == "__main__":
    # This block demonstrates how to use the RobotConnection class.
    # Replace 'COM4' with the actual port your robot is on.
    PORT = 'COM4'
    
    # Create an instance of the connection manager
    robot_conn = RobotConnection()
    
    # Connect to the robot
    if robot_conn.connect(PORT):
        try:
            # Perform the initial handshake
            robot_conn.wait_for_setup_confirmation()
            time.sleep(1)

            # Example: Send a command to move a servo
            print("Sending command to move servo 1 to 90 degrees.")
            # This corresponds to the old 'four_function'
            robot_conn.send_command('servo', 1, 90, 500) 
            time.sleep(2)

            print("Sending command to move servo 1 to 45 degrees.")
            robot_conn.send_command('servo', 1, 45, 500)
            time.sleep(2)

        except Exception as e:
            print(f"An error occurred during communication: {e}")
        finally:
            # Always ensure the connection is closed
            robot_conn.disconnect()

