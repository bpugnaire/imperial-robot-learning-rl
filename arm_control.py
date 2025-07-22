import json, serial, time

def send(port, *args):
    line = json.dumps({"start": list(args)}) + "\n"
    port.write(line.encode("utf-8"))

def read_line(port, timeout=1.0):
    port.timeout = timeout
    data = port.readline()
    return data.decode("utf-8", errors="ignore").strip() if data else None

PORT = "COM10"
BAUD = 115200

# Your 5 servos: channels 0–4, mapped to MCU pins (change these!)
PINS = [9, 6, 5, 3, 11]

ANGLES = [90, 90, 90, 90, 90]   # start angles

with serial.Serial(PORT, BAUD, timeout=1, write_timeout=1) as ser:
    # wait for device to reply to 'setup'
    while True:
        send(ser, "setup")
        if read_line(ser): break
        time.sleep(0.1)

    # attach all 5
    for ch, pin in enumerate(PINS):
        send(ser, "servo_attach", ch, pin)

    # move one servo
    send(ser, "servo_write", 0, 120)

    # move all 5
    for ch, angle in enumerate(ANGLES):
        send(ser, "servo_write", ch, angle)

    # optional readback
    send(ser, "get_val", 42)
    print("reply:", read_line(ser))


