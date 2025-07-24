import time
from src.hardware.python_controller.robot_interface import Robot
import math

def degree_to_radian(deg):
    return math.pi * deg / 180

if __name__ == "__main__":

    port = "/dev/ttyACM0"
    bounds = [0.4, 1.1]
    robot = Robot(port=port, join_idx=2, bounds=bounds)
    robot.reset_position()
    time.sleep(1)
    robot.move_joint(-20)
    time.sleep(1)
    robot.move_joint(-20)