# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 16:47:42 2025

@author: khelvig
"""
import cv2
import numpy as np
import pyrealsense2 as rs

def detect_red_objects_frame(frame, circularity_threshold=0.75, border_margin=20):
    """
    Process a BGR image frame and return the annotated image and detection list.
    """
    frame = cv2.resize(frame, (640, 480))
    h, w = frame.shape[:2]

    # Convert to HSV and mask red regions
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Noise reduction using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find external contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = frame.copy()
    results = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 220:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        if (x <= border_margin or y <= border_margin or
            x + cw >= w - border_margin or y + ch >= h - border_margin):
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) == 3:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])

        if circularity >= circularity_threshold:
            label = 'Circle'
            cv2.circle(output, (cx, cy), 5, (255, 0, 0), -1)
        else:
            label = 'Cube'
            if len(approx) == 4:
                for p in approx:
                    px, py = p[0]
                    cv2.circle(output, (px, py), 5, (0, 255, 0), -1)
            else:
                roi = output[y:y+ch, x:x+cw]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
                corners = cv2.cornerHarris(gray, blockSize=4, ksize=5, k=0.04)
                corners = cv2.dilate(corners, None)
                thresh = 0.1 * corners.max()
                ys, xs = np.where(corners > thresh)
                for ix, iy in zip(xs, ys):
                    cv2.circle(output, (x+ix, y+iy), 3, (0, 255, 0), -1)

        cv2.drawContours(output, [cnt], -1, (0, 255, 255), 2)
        cv2.putText(output, label, (cx-20, cy-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        results.append({'label': label,
                        'circularity': circularity,
                        'centroid': (cx, cy)})

    return output, results


if __name__ == '__main__':
    # Choose mode: 'image', 'webcam', or 'realsense'
    mode = 'webcam'
    image_path = None            # set path if mode == 'image'
    circularity_threshold = 0.75

    if mode == 'image':
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: cannot load image {image_path}")
            exit(1)
        annotated, dets = detect_red_objects_frame(img, circularity_threshold)
        print("Detections:")
        for d in dets:
            print(f"{d['label']} - circ: {d['circularity']:.2f} at {d['centroid']}")
        cv2.imshow('Image Result', annotated)
        cv2.waitKey(0)

    elif mode == 'webcam':
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: cannot open webcam")
            exit(1)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated, _ = detect_red_objects_frame(frame, circularity_threshold)
            cv2.imshow('Webcam Detection', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()

    elif mode == 'realsense':
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        pipeline.start(config)
        try:
            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                annotated, _ = detect_red_objects_frame(color_image, circularity_threshold)
                cv2.imshow('RealSense Detection', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            pipeline.stop()

    cv2.destroyAllWindows()

