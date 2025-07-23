# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 15:53:42 2025

@author: khelvig
"""
import cv2
import numpy as np
import math
import pyrealsense2 as rs
import matplotlib.pyplot as plt

# Path or None; MODE: 'image', 'webcam', 'realsense'
IMAGE_PATH = r"C:\Users\khelvig\Downloads\test.png"
MODE = 'webcam'
MIN_BLOB_AREA = 300  # adjust for smaller blobs

# Pink HSV ranges (expanded saturation/value)
LOWER_PINK1 = np.array([145, 50, 50])
UPPER_PINK1 = np.array([180, 255, 255])
LOWER_PINK2 = np.array([0, 50, 50])
UPPER_PINK2 = np.array([10, 255, 255])

# Enhance low-light: CLAHE on L channel + histogram equalization on V

def enhance_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Detect pink blob and return fitted ellipse parameters

def detect_pink_blob(frame, debug=False):
    proc = enhance_frame(frame)
    if debug:
        cv2.imshow('enhanced', proc)

    hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, LOWER_PINK1, UPPER_PINK1),
        cv2.inRange(hsv, LOWER_PINK2, UPPER_PINK2)
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    if debug:
        cv2.imshow('mask', mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_BLOB_AREA or len(largest) < 5:
        return None

    ellipse = cv2.fitEllipse(largest)
    # ellipse: ((cx, cy), (axis1, axis2), raw_angle)
    return ellipse

# Draw ellipse, axes (using raw_angle), and angle relative to horizontal

def draw_ellipse_with_axes(frame, ellipse):
    out = frame.copy()
    (cx, cy), (axis1, axis2), raw_angle = ellipse
    # Draw the ellipse outline
    cv2.ellipse(out, ellipse, (0, 255, 0), 2)

    # Determine major/minor axis lengths and orientation
    if axis1 >= axis2:
        major_len = axis1 / 2
        minor_len = axis2 / 2
        angle_rad = math.radians(raw_angle)
    else:
        major_len = axis2 / 2
        minor_len = axis1 / 2
        # raw_angle corresponds to axis1; add 90° to align with axis2
        angle_rad = math.radians(raw_angle + 90)

    # Compute unit vector along major axis
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    # Endpoints for major axis
    pt1 = (int(cx + dx * major_len), int(cy + dy * major_len))
    pt2 = (int(cx - dx * major_len), int(cy - dy * major_len))
    # Perpendicular unit vector for minor axis
    perp_dx = -dy
    perp_dy = dx
    pt3 = (int(cx + perp_dx * minor_len), int(cy + perp_dy * minor_len))
    pt4 = (int(cx - perp_dx * minor_len), int(cy - perp_dy * minor_len))

    # Draw axes: major in blue, minor in red
    cv2.line(out, pt1, pt2, (255, 0, 0), 2)
    cv2.line(out, pt3, pt4, (0, 0, 255), 2)

    # Compute angle in degrees relative to horizontal (0° = rightwards)
    angle_deg = math.degrees(angle_rad)
    # Normalize to [0,180)
    angle_deg = angle_deg % 180
    # Convert to acute: [0,90]
    if angle_deg > 90:
        angle_deg = 180 - angle_deg

    cv2.putText(out, f"Angle: {angle_deg:.1f}°", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return out

# Main loop variants

def process_image(path):
    frame = cv2.imread(path)
    frame = cv2.resize(frame, (640, 480))
    ellipse = detect_pink_blob(frame, debug=True)
    out = draw_ellipse_with_axes(frame, ellipse) if ellipse else frame
    cv2.imshow('Result', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        ellipse = detect_pink_blob(frame)
        out = draw_ellipse_with_axes(frame, ellipse) if ellipse else frame
        cv2.imshow('Webcam', out)
        if cv2.waitKey(1) in [27, ord('q')]:
            break
    cap.release()
    cv2.destroyAllWindows()


def process_realsense():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(cfg)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            cf = frames.get_color_frame()
            if not cf:
                continue
            frame = np.asanyarray(cf.get_data())
            frame = cv2.resize(frame, (640, 480))
            ellipse = detect_pink_blob(frame)
            out = draw_ellipse_with_axes(frame, ellipse) if ellipse else frame
            cv2.imshow('RealSense', out)
            if cv2.waitKey(1) in [27, ord('q')]:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    if MODE == 'image':
        process_image(IMAGE_PATH)
    elif MODE == 'webcam':
        process_webcam()
    elif MODE == 'realsense':
        process_realsense()

