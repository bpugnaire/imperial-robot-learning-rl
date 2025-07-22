#!/usr/bin/env python3
"""
detect_board_port.py

Steps:
 1. Run with the board CONNECTED.
 2. When prompted, unplug the board and press Enter.
 3. Plug it back in and press Enter again.

The script will report the most likely port (Windows: COMx, Linux: /dev/ttyUSBx or /dev/ttyACMx, macOS: /dev/tty.usbmodem*).
"""
import time
from typing import Set
try:
    from pyserial.tools import list_ports
except ImportError:
    raise SystemExit("pip install pyserial")

def snapshot_ports() -> Set[str]:
    return {p.device for p in list_ports.comports()}

def pretty_list(s: Set[str]) -> str:
    return ", ".join(sorted(s)) if s else "(none)"

def main():
    print("Step 1/3: Taking initial snapshot (board should be connected).")
    first = snapshot_ports()
    print("Currently seen ports:", pretty_list(first))
    input("Step 2/3: Unplug the board, then press Enter... ")
    second = snapshot_ports()
    print("Ports after unplug:", pretty_list(second))

    disappeared = first - second
    if not disappeared:
        print("No ports disappeared. Maybe you unplugged the wrong thing? Continuing anyway.")
    else:
        print("These ports disappeared:", pretty_list(disappeared))

    input("Step 3/3: Plug the board back in, then press Enter... ")
    third = snapshot_ports()
    print("Ports after replug:", pretty_list(third))

    appeared = third - second
    if not appeared:
        print("No new ports appeared. Try again.")
        return

    print("These ports appeared:", pretty_list(appeared))

    # Heuristic: if exactly one port both disappeared and reappeared, that’s our guy.
    candidate = (disappeared & appeared) or appeared
    if len(candidate) == 1:
        port = next(iter(candidate))
        print(f"\n✅ Detected board on: {port}")
    else:
        print("\n⚠️ Multiple candidates. Pick one manually from this list:")
        for p in sorted(candidate):
            print("  -", p)

    # Optional: try opening and pinging the device here if you have a known handshake.
    # (Uncomment and adapt)
    """
    import serial, json
    def send(ser, *args):
        ser.write((json.dumps({'start': list(args)}) + '\\n').encode('utf-8'))
    with serial.Serial(port, 115200, timeout=1) as ser:
        send(ser, 'setup')
        print('Response:', ser.readline())
    """

if __name__ == "__main__":
    main()


