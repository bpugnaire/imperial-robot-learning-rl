#!/usr/bin/env python3
# dance.py

import argparse
import time
import numpy as np
from bus import FeetechBus
import threading
import sys
import termios
import tty

# --- Global flag to signal the listener thread to stop ---
stop_movement = False

# --- Safety First: Define Joint Limits (in radians) ---
# The limits below are calculated from the provided calibration data to match
# the physical range of the servos.
# Formula used: rad = (raw_encoder_value - 2048 - homing_offset) * (2 * pi / 4096)
# [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
JOINT_LIMITS = {
    'shoulder_pan':  (-1.7227, 1.8303),
    'shoulder_lift': (-1.9513, 1.7410),
    'elbow_flex':    (-1.8254, 1.5846),
    'wrist_flex':    (-1.7549, 1.8939),
    'wrist_roll':    (-2.8420, 3.0202),
    'gripper':       (0.0031, 2.2197),
}

# Define the names of the joints in order from base to gripper.
JOINT_NAMES = list(JOINT_LIMITS.keys())


def listen_for_key_press():
    """
    Waits for a single key press and sets a global flag to stop movement.
    """
    global stop_movement
    print("Dance started. Press any key to stop.")
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        sys.stdin.read(1)
        stop_movement = True
        print("\nKey pressed. Finishing current move and stopping...")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def send_waypoints(bus, current_pos, target_pos, duration=1.5, steps=40):
    """
    Sends a series of intermediate waypoints to smoothly move the arm.
    """
    global stop_movement
    if stop_movement:
        return
        
    delay = duration / steps
    for i in range(1, steps + 1):
        # Check if a stop is requested during the move
        if stop_movement:
            break
        intermediate_pos = current_pos + (target_pos - current_pos) * (i / steps)
        bus.set_qpos(intermediate_pos)
        time.sleep(delay)


def main():
    """
    Main function to run the dance routine.
    """
    global stop_movement
    ap = argparse.ArgumentParser(description="Make the robot arm perform a little dance.")
    ap.add_argument("--port", default="/dev/ttyACM0", help="The serial port for the Feetech bus.")
    ap.add_argument("--calib", default="so101_calibration.json", help="Path to the calibration file.")
    ap.add_argument("--ids", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6], help="Bus IDs of the servos.")
    args = ap.parse_args()

    bus = FeetechBus(args.port, args.ids, calib_file=args.calib)
    
    # --- The Dance Routine ---
    # Each pose is a list of joint angles in radians, following JOINT_NAMES order.
    # All poses should be within the JOINT_LIMITS defined above.
    home_pose = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)

    dance_routine = [
        # 1. "Wave"
        {'pose': np.array([0, 0.5, 0.8, 0.5, -1.0, 0.5]), 'duration': 1.5},
        {'pose': np.array([0, 0.5, 0.8, 0.5, 1.0, 0.5]), 'duration': 1.0},
        {'pose': np.array([0, 0.5, 0.8, 0.5, -1.0, 0.5]), 'duration': 1.0},
        {'pose': np.array([0, 0.5, 0.8, 0.5, 1.0, 0.5]), 'duration': 1.0},
        
        # 2. "Peek-a-boo"
        {'pose': np.array([0, -0.5, -1.2, -1.0, 0, 0.8]), 'duration': 2.0}, # Hide
        {'pose': np.array([0.8, 0.4, 0.5, 0, 0, 0.2]), 'duration': 1.5},   # Peek left
        {'pose': np.array([-0.8, 0.4, 0.5, 0, 0, 0.2]), 'duration': 2.0},  # Peek right
        
        # 3. "Shimmy"
        {'pose': np.array([-0.8, 0.2, 0.5, 0.8, 0, 0.5]), 'duration': 1.5},
        {'pose': np.array([0.8, 0.2, 0.5, 0.8, 0, 0.5]), 'duration': 1.5},
        
        # 4. "Bow"
        {'pose': np.array([0, 0.8, 1.5, 1.0, 0, 0.8]), 'duration': 2.5},
        {'pose': home_pose, 'duration': 2.0} # Return to home
    ]

    # Start the keyboard listener
    listener_thread = threading.Thread(target=listen_for_key_press, daemon=True)
    listener_thread.start()

    try:
        bus.set_torque(True)
        time.sleep(0.1)

        # Start at the home position
        current_qpos = bus.get_qpos()
        print("Moving to home position...")
        send_waypoints(bus, current_qpos, home_pose, duration=2.0)
        time.sleep(1)

        # Loop the dance
        while not stop_movement:
            for move in dance_routine:
                if stop_movement:
                    break
                target_qpos = move['pose']
                duration = move['duration']
                
                # Get current position before each move
                current_qpos = bus.get_qpos()
                
                print(f"Performing move: {target_qpos}")
                send_waypoints(bus, current_qpos, target_qpos, duration=duration)
            
            if not stop_movement:
                print("Routine finished. Repeating in 3 seconds...")
                time.sleep(3)

    finally:
        print("Cleaning up...")
        # Go to a safe, neutral position before turning off torque
        current_qpos = bus.get_qpos()
        send_waypoints(bus, current_qpos, home_pose, duration=2.0)
        bus.set_torque(False)
        bus.disconnect()
        print("Disconnected.")

if __name__ == "__main__":
    main()
