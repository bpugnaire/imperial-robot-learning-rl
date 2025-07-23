import time
from robot_interface import Robot
import math

def degree_to_radian(deg):
    return math.pi * deg / 180

if __name__ == "__main__":

    port = "/dev/ttyACM0"

    robot = Robot(port=port, join_idx=3)
    robot.reset_position()
    time.sleep(1)
    robot.move_joint(degree_to_radian(5))
    time.sleep(1)
    robot.move_joint(degree_to_radian(5))