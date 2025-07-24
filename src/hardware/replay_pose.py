#!/usr/bin/env python3
# replay_pose.py

import argparse
import json
import time
import numpy as np
from src.hardware.python_controller.bus import FeetechBus

# Define the names of the joints in order from base to gripper.
# This is used to ensure the joint angles from the file are applied correctly.
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

def send_waypoints(bus, current_pos, target_pos, duration=2.0, steps=100):
    """
    Sends a series of intermediate waypoints to smoothly move the arm.

    Args:
        bus: The FeetechBus instance.
        current_pos (np.ndarray): The starting position of the arm.
        target_pos (np.ndarray): The destination position of the arm.
        duration (float): The total time the movement should take in seconds.
        steps (int): The number of intermediate steps to generate.
    """
    print(f"Moving from {current_pos} to {target_pos}...")
    delay = duration / steps
    for i in range(1, steps + 1):
        # Calculate the intermediate position using linear interpolation.
        intermediate_pos = current_pos + (target_pos - current_pos) * (i / steps)
        bus.set_qpos(intermediate_pos)
        time.sleep(delay)
    # print("Movement complete.")

def main():
    """
    Main function to load a pose and move the arm.
    """
    # Set up command-line argument parsing.
    ap = argparse.ArgumentParser(
        description="Load a saved pose from a JSON file and move the arm to that position."
    )
    ap.add_argument("filename",
                    help="Path to the input JSON file containing the target pose.")
    ap.add_argument("--port", default="/dev/ttyACM0",
                    help="The serial port for the Feetech bus.")
    ap.add_argument("--calib", default="so101_calibration.json",
                    help="Path to the calibration file.")
    ap.add_argument("--ids", nargs="+", type=int,
                    default=[1, 2, 3, 4, 5, 6],
                    help="Bus IDs of the servos, ordered from base to gripper.")
    args = ap.parse_args()

    # Initialize the connection to the robot arm's servos.
    # The calibration file is necessary to correctly map angles to servo positions.
    bus = FeetechBus(args.port, args.ids, calib_file=args.calib)

    try:
        # --- Load the target pose from the file ---
        # print(f"Loading pose from {args.filename}...")
        with open(args.filename, 'r') as f:
            pose_data = json.load(f)

        # Create a numpy array for the target position, ensuring the order of
        # angles matches the JOINT_NAMES list.
        target_qpos = np.array([pose_data[name] for name in JOINT_NAMES], dtype=np.float32)

        # --- Move the arm ---
        # Enable torque to allow the servos to hold their position and move.
        # print("Enabling torque...")
        bus.set_torque(True)
        time.sleep(0.1) # Give a moment for the servos to engage.

        # Get the arm's current position to use as the starting point.
        current_qpos = bus.get_qpos()
        # print(f"Current position: {np.round(current_qpos, 3)}")
        # print(f"Target position:  {np.round(target_qpos, 3)}")

        # Send the arm to the target position.
        send_waypoints(bus, current_qpos, target_qpos)

        # Wait for a moment at the target position.
        time.sleep(1)

    finally:
        # This block ensures that the connection is always closed properly.
        # print("Disconnecting from bus.")
        bus.disconnect()


if __name__ == "__main__":
    main()

