import serial
import time

# --- Use the correct serial port for your system ---
try:
    # Replace 'COM4' with your port and 9600 with the baud rate
    ser = serial.Serial('COM4', 9600, timeout=1) 
    time.sleep(2) # Wait for the connection to establish
except serial.SerialException as e:
    print(f"Error: Could not open serial port. {e}")
    exit()

def move_servo(servo_num, angle):
    """Sends a command to move a servo to a specific angle."""
    if not (1 <= servo_num <= 5 and 0 <= angle <= 180):
        print("Error: Invalid servo number or angle.")
        return
    
    # This command format is based on common servo control protocols
    command = f"#{servo_num}P{angle}T100\r\n"
    ser.write(command.encode())
    print(f"Moving servo {servo_num} to {angle} degrees.")

# --- Main execution ---
try:
    # Example: Center all servos
    for i in range(1, 6):
        move_servo(i, 90)
        time.sleep(0.5)

finally:
    ser.close()
    print("Serial connection closed.")