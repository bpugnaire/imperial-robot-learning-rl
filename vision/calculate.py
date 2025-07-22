# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 16:03:37 2025
Estimate angle with LSD 
@author: khelvig
"""
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt 

def plot_detected_segments(img, segments, window_name='Segments detected'):
    """plotting"""
    vis = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in segments:
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Red color in BGR
    cv2.imshow(window_name, vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def filter_pink_regions(img):
    """Filter to keep only pink regions in the image"""
    # Convert BGR to HSV for better color filtering
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for pink color in HSV (less aggressive)
    lower_pink = np.array([130, 50, 30])
    upper_pink = np.array([180, 255, 255])
    
    # Create mask for pink regions
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    
    # Apply mask to original image
    filtered_img = cv2.bitwise_and(img, img, mask=mask)
    
    return filtered_img, mask

def estimate_opening_angle(image_path, show_segments=False, plot_steps=True):
    # 1. image + blue filtering
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 480))
    
    if plot_steps:
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        plt.show()
    
    # Filter for pink regions only
    filtered_img, pink_mask = filter_pink_regions(img)
    
    if plot_steps:
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
        plt.title('Pink Filtered Image')
        plt.axis('off')
        plt.show()
        
        plt.figure(figsize=(8, 6))
        plt.imshow(pink_mask, cmap='gray')
        plt.title('Pink Mask')
        plt.axis('off')
        plt.show()
    
    # Convert to grayscale (the pink mask can be used directly)
    gray = pink_mask
    
    # increase contrast for easier edge detection 
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(3, 3))
    gray = clahe.apply(gray.astype(np.uint8))
    
    if plot_steps:
        plt.figure(figsize=(8, 6))
        plt.imshow(gray, cmap='gray')
        plt.title('CLAHE Enhanced')
        plt.axis('off')
        plt.show()
    
    # 2. LSD line segment detection (opencv)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray)[0]  

    # 3. parameter for lsd : isolate right size lines ? 
    min_length = 20
    segs = []
    for l in lines:
        x1,y1,x2,y2 = l[0]
        if np.hypot(x2-x1, y2-y1) > min_length:
            segs.append((x1,y1,x2,y2))

    if plot_steps:
        # Plot all detected segments
        plt.figure(figsize=(8, 6))
        vis = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2RGB)
        for x1, y1, x2, y2 in segs:
            plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2)
        plt.imshow(gray, cmap='gray', alpha=0.7)
        plt.title(f'Filtered Segments (n={len(segs)})')
        plt.axis('off')
        plt.show()

    if show_segments:
        plot_detected_segments(gray, segs)

    # Filter to keep only the longest segment
    if segs:
        lengths = [np.hypot(x2-x1, y2-y1) for x1, y1, x2, y2 in segs]
        longest_idx = np.argmax(lengths)
        segs = [segs[longest_idx]]
        
        # Add horizontal line (horizon)
        img_height, img_width = gray.shape
        horizon_y = img_height // 2
        horizon_line = (0, horizon_y, img_width-1, horizon_y)
        segs.append(horizon_line)
        
        if plot_steps:
            plt.figure(figsize=(8, 6))
            plt.imshow(gray, cmap='gray', alpha=0.7)
            # Plot longest segment in red
            x1, y1, x2, y2 = segs[0]
            plt.plot([x1, x2], [y1, y2], 'r-', linewidth=3, label='Longest segment')
            # Plot horizon in green
            x1, y1, x2, y2 = segs[1]
            plt.plot([x1, x2], [y1, y2], 'g--', linewidth=2, label='Horizon')
            plt.title('Longest Segment + Horizon')
            plt.legend()
            plt.axis('off')
            plt.show()

    # 4. estimate orientation (trigonometry stuffs)
    angles = [math.atan2(y2-y1, x2-x1) for x1,y1,x2,y2 in segs]
    data = np.array(angles, dtype=np.float32).reshape(-1,1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 5. FitLine cluster
    line_params = []
    cluster_colors = ['red', 'blue']
    
    if plot_steps:
        plt.figure(figsize=(8, 6))
        plt.imshow(gray, cmap='gray', alpha=0.7)
        
    for k in range(2):
        pts = []
        for i, seg in enumerate(segs):
            if labels[i] == k:
                x1,y1,x2,y2 = seg
                pts.extend([[x1,y1],[x2,y2]])
                if plot_steps:
                    plt.plot([x1, x2], [y1, y2], color=cluster_colors[k], linewidth=2, alpha=0.8)
        vx,vy,_,_ = cv2.fitLine(np.array(pts), cv2.DIST_L2,0,0.01,0.01).flatten()
        line_params.append((vx, vy))

    if plot_steps:
        plt.title('Clustered Segments')
        plt.axis('off')
        plt.show()

    # 6. angle
    (v1x,v1y),(v2x,v2y) = line_params
    cosang = abs(v1x*v2x + v1y*v2y) / (math.hypot(v1x,v1y)*math.hypot(v2x,v2y))
    angle_deg = math.degrees(math.acos(np.clip(cosang, -1.0, 1.0)))
    return angle_deg

if __name__ == "__main__":
    path = r"images/test.png"
    # path = r"C:\Users\khelvig\Downloads\demo.jpg"
    angle = estimate_opening_angle(path, show_segments=True, plot_steps=True)
    print(f"Opening angle : {angle:.2f}°")