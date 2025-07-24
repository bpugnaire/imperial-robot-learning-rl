#!/usr/bin/env python3
# exercise_joint.py

import argparse
import json
import time
import numpy as np
from hardware.python_controller.bus import FeetechBus
import threading
import sys
import termios
import tty

# --- Global flag to signal the listener thread to stop ---
stop_movement = False

# Define the names of the joints in order from base to gripper.
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
# The index of the elbow joint in the JOINT_NAMES list.
ELBOW_JOINT_INDEX = 2

def listen_for_key_press():
    """
    Waits for a single key press and sets a global flag.
    NOTE: This function is for Unix-like systems (Linux, macOS).
    """
    global stop_movement
    print("Movement started. Press any key to stop.")
    
    # Get the original terminal settings
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        # Set the terminal to raw mode (cbreak)
        tty.setcbreak(sys.stdin.fileno())
        # Wait for a single character of input
        sys.stdin.read(1)
        stop_movement = True
        print("\nKey pressed. Stopping movement...")
    finally:
        # Always restore the original terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def send_waypoints(bus, current_pos, target_pos, duration=1.5, steps=40):
    """
    Sends a series of intermediate waypoints to smoothly move the arm.
    Stops immediately if the global stop_movement flag is set.

    Args:
        bus: The FeetechBus instance.
        current_pos (np.ndarray): The starting position of the arm.
        target_pos (np.ndarray): The destination position of the arm.
        duration (float): The total time the movement should take in seconds.
        steps (int): The number of intermediate steps to generate.
    """
    global stop_movement
    delay = duration / steps
    for i in range(1, steps + 1):
        if stop_movement:
            return # Exit early if a key has been pressed
            
        # Calculate the intermediate position using linear interpolation.
        intermediate_pos = current_pos + (target_pos - current_pos) * (i / steps)
        bus.set_qpos(intermediate_pos)
        time.sleep(delay)

def main():
    """
    Main function to exercise the elbow joint.
    """
    # Set up command-line argument parsing.
    ap = argparse.ArgumentParser(
        description="Move the elbow joint between an upper and lower bound until a key is pressed."
    )
    ap.add_argument("--lower_bound", type=float, default=0.4,
                    help="The lower bound for the elbow joint (in radians).")
    ap.add_argument("--upper_bound", type=float, default=1.3,
                    help="The upper bound for the elbow joint (in radians).")
    ap.add_argument("--port", default="/dev/ttyACM0",
                    help="The serial port for the Feetech bus.")
    ap.add_argument("--calib", default="so101_calibration.json",
                    help="Path to the calibration file.")
    ap.add_argument("--ids", nargs="+", type=int,
                    default=[1, 2, 3, 4, 5, 6],
                    help="Bus IDs of the servos, ordered from base to gripper.")
    args = ap.parse_args()

    # Initialize the connection to the robot arm's servos.
    bus = FeetechBus(args.port, args.ids, calib_file=args.calib)

    # Start the keyboard listener in a separate thread
    listener_thread = threading.Thread(target=listen_for_key_press, daemon=True)
    listener_thread.start()

    try:
        # Enable torque to allow the servos to hold their position and move.
        bus.set_torque(True)
        time.sleep(0.1) # Give a moment for the servos to engage.

        # Get the arm's current position. The other joints will hold this position.
        initial_qpos = bus.get_qpos()
        print(f"Holding other joints at: {np.round(initial_qpos, 3)}")

        # Define the target positions for the elbow joint
        pos_lower = initial_qpos.copy()
        pos_lower[ELBOW_JOINT_INDEX] = args.lower_bound

        pos_upper = initial_qpos.copy()
        pos_upper[ELBOW_JOINT_INDEX] = args.upper_bound
        
        # First, move to the starting lower bound position
        print(f"Moving to starting position (elbow at {args.lower_bound:.2f} rad)...")
        send_waypoints(bus, initial_qpos, pos_lower)

        # Loop the movement until the listener thread sets the stop flag
        current_pos = pos_lower
        while not stop_movement:
            # Move to the upper bound
            print(f"Moving to upper bound ({args.upper_bound:.2f} rad)...")
            send_waypoints(bus, current_pos, pos_upper)
            current_pos = pos_upper
            if stop_movement: break

            # Move back to the lower bound
            print(f"Moving to lower bound ({args.lower_bound:.2f} rad)...")
            send_waypoints(bus, current_pos, pos_lower)
            current_pos = pos_lower
            if stop_movement: break
        
    finally:
        # This block ensures that cleanup code runs reliably.
        print("Cleaning up...")
        print("Returning to initial position...")
        send_waypoints(bus, bus.get_qpos(), initial_qpos)
        print("Returned to initial position. Torque remains enabled.")
        bus.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
