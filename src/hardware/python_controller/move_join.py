import time

def move_join(bus, current_pos, target_pos, duration=1.0, steps=100):
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
    print("Movement complete.")