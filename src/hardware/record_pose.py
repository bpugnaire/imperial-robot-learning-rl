#!/usr/bin/env python3
# record_pose.py

import argparse
import json
from src.hardware.python_controller.bus import FeetechBus

# Define the names of the joints in order from base to gripper.
# This should match the order of the servo IDs provided.
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

def main():
    """
    Main function to run the pose recording process.
    """
    # Set up command-line argument parsing.
    ap = argparse.ArgumentParser(
        description="Move a 6-DOF arm to a desired pose and save the joint configuration."
    )
    ap.add_argument("--filename", default="start_pose.json",
                    help="Path to the output JSON file where the pose will be saved.")
    ap.add_argument("--port", default="/dev/ttyACM0",
                    help="The serial port for the Feetech bus.")
    ap.add_argument("--ids", nargs="+", type=int,
                    default=[1, 2, 3, 4, 5, 6],
                    help="Bus IDs of the servos, ordered from base to gripper.")
    args = ap.parse_args()

    # Ensure the number of joint names matches the number of IDs.
    if len(JOINT_NAMES) != len(args.ids):
        raise ValueError("The number of joint names must match the number of servo IDs.")

    # Initialize the connection to the robot arm's servos.
    bus = FeetechBus(args.port, args.ids)

    try:
        # Disable torque on all servos to allow for manual positioning.
        print("Disabling torque to allow manual positioning...")
        bus.set_torque(False)

        # Prompt the user to move the arm into the desired position.
        input("\nMove the arm to the desired pose, then press ENTER to record...")

        # Read the current position of each servo.
        # We assume get_qpos() returns calibrated joint angles (e.g., in radians).
        print("Reading current joint positions...")
        pose_angles = bus.get_qpos()

        # Create a dictionary that maps joint names to their recorded angles.
        pose_data = dict(zip(JOINT_NAMES, pose_angles))

        print("\nRecorded Pose:")
        for name, angle in pose_data.items():
            # Print each joint's angle, formatted to 4 decimal places.
            print(f"  {name}: {angle:.4f}")

        # ----------- Save the pose data to a file -----------------------------
        print(f"\nSaving pose to {args.filename}...")
        with open(args.filename, "w") as f:
            json.dump(pose_data, f, indent=4)
        print("Pose saved successfully.")

    finally:
        # This block ensures that cleanup code runs even if an error occurs.
        print("\nCleaning up...")
        try:
            # It's good practice to ensure torque is off before disconnecting.
            bus.set_torque(False)
        finally:
            # Disconnect from the bus to release the serial port.
            bus.disconnect()
            print("Disconnected.")


if __name__ == "__main__":
    main()

