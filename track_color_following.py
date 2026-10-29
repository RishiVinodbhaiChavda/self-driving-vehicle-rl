"""
Track Color Following System
Specifically designed for the driving game with gray road and green grass

STRATEGY:
1. Detect road color (gray/brown) → Binary mask (white)
2. Detect grass (green) → Mark as off-track (black)
3. Follow ONLY the white (road) areas
4. If goes on grass → Press 'R' to reset + penalty

This is much simpler and more reliable than edge detection!
"""

import numpy as np
from PIL import ImageGrab
import cv2
import time
from direction_keys import ReleaseKey, PressKey, W, A, S, D
import ctypes


# Add R key for reset
R_KEY = 0x13


def press_r_key():
    """Press R key to reset in game"""
    extra = ctypes.c_ulong(0)
    
    class KeyBdInput(ctypes.Structure):
        _fields_ = [("wVk", ctypes.c_ushort),
                   ("wScan", ctypes.c_ushort),
                   ("dwFlags", ctypes.c_ulong),
                   ("time", ctypes.c_ulong),
                   ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]
    
    class Input_I(ctypes.Union):
        _fields_ = [("ki", KeyBdInput)]
    
    class Input(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]
    
    # Press
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, R_KEY, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    
    time.sleep(0.05)
    
    # Release
    ii_.ki = KeyBdInput(0, R_KEY, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def calibrate_road_color(sample_image):
    """
    Analyze the screenshot to find exact road color ranges
    Based on your screenshot: gray/brown road
    """
    # Convert to HSV
    hsv = cv2.cvtColor(sample_image, cv2.COLOR_BGR2HSV)
    
    # Sample the center bottom area (definitely road)
    h, w = sample_image.shape[:2]
    road_sample = hsv[int(h*0.7):int(h*0.85), int(w*0.4):int(w*0.6)]
    
    # Calculate statistics
    h_mean, s_mean, v_mean = np.mean(road_sample.reshape(-1, 3), axis=0)
    h_std, s_std, v_std = np.std(road_sample.reshape(-1, 3), axis=0)
    
    print("=" * 60)
    print("ROAD COLOR CALIBRATION")
    print("=" * 60)
    print(f"Road HSV Mean: H={h_mean:.1f}, S={s_mean:.1f}, V={v_mean:.1f}")
    print(f"Road HSV Std:  H={h_std:.1f}, S={s_std:.1f}, V={v_std:.1f}")
    
    # Define ranges (mean ± 2*std for 95% coverage)
    lower_road = np.array([
        max(0, h_mean - 2*h_std),
        max(0, s_mean - 2*s_std),
        max(0, v_mean - 2*v_std)
    ]).astype(np.uint8)
    
    upper_road = np.array([
        min(180, h_mean + 2*h_std),
        min(255, s_mean + 2*s_std),
        min(255, v_mean + 2*v_std)
    ]).astype(np.uint8)
    
    print(f"\nRecommended HSV ranges:")
    print(f"  Lower: {lower_road}")
    print(f"  Upper: {upper_road}")
    print("=" * 60)
    
    return lower_road, upper_road


def process_track_binary(orig_img, show_color_info=False):
    """
    Convert image to binary: WHITE = road, BLACK = off-track
    
    Based on your game screenshot:
    - Road: Gray/brown (low saturation, medium value)
    - Grass: Green (medium saturation, medium-high value)
    - Sky: Blue/white (varies)
    """
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # ROAD DETECTION (Gray/Brown road from your screenshot)
    # Gray roads: Low saturation, medium value
    # H: any (gray has no dominant hue)
    # S: 0-60 (low saturation = grayish)
    # V: 40-130 (medium brightness)
    lower_road = np.array([0, 0, 40])
    upper_road = np.array([180, 60, 130])
    road_mask = cv2.inRange(hsv, lower_road, upper_road)
    
    # GRASS DETECTION (Green grass - to explicitly mark as off-track)
    # H: 35-85 (green hues)
    # S: 30-255 (some saturation)
    # V: 40-255 (visible green)
    lower_grass = np.array([35, 30, 40])
    upper_grass = np.array([85, 255, 255])
    grass_mask = cv2.inRange(hsv, lower_grass, upper_grass)
    
    # WHITE LANE MARKINGS (to help with centering)
    # Very high value, low saturation
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Combine: Road + White markings = drivable
    drivable_mask = cv2.bitwise_or(road_mask, white_mask)
    
    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    drivable_mask = cv2.morphologyEx(drivable_mask, cv2.MORPH_CLOSE, kernel)
    drivable_mask = cv2.morphologyEx(drivable_mask, cv2.MORPH_OPEN, kernel)
    
    # Create binary output: WHITE = road, BLACK = off-track
    binary_track = drivable_mask.copy()
    
    # Optional: Show color-coded version
    if show_color_info:
        color_visual = np.zeros_like(orig_img)
        color_visual[road_mask > 0] = [128, 128, 128]  # Gray for road
        color_visual[grass_mask > 0] = [0, 255, 0]      # Green for grass
        color_visual[white_mask > 0] = [255, 255, 255]  # White for markings
        return binary_track, color_visual
    
    return binary_track, None


def check_if_off_track(binary_mask, check_threshold=0.3):
    """
    Check if car is off the track
    
    Args:
        binary_mask: Binary image (white=road, black=off-track)
        check_threshold: If less than this % is road, consider off-track
        
    Returns:
        is_off_track: Boolean
        road_percentage: Float (0-1)
    """
    height, width = binary_mask.shape
    
    # Check bottom-center region (where car is)
    car_region_y_start = int(height * 0.7)
    car_region_y_end = int(height * 0.9)
    car_region_x_start = int(width * 0.4)
    car_region_x_end = int(width * 0.6)
    
    car_region = binary_mask[car_region_y_start:car_region_y_end,
                             car_region_x_start:car_region_x_end]
    
    # Calculate percentage of white (road) pixels
    total_pixels = car_region.size
    road_pixels = np.sum(car_region > 0)
    road_percentage = road_pixels / total_pixels
    
    # Off track if less than threshold
    is_off_track = road_percentage < check_threshold
    
    return is_off_track, road_percentage


def calculate_track_center(binary_mask):
    """
    Calculate where the center of the track is
    Returns offset from image center (-1 to +1)
    """
    height, width = binary_mask.shape
    
    # Look ahead region (upper portion where car should aim)
    lookahead_y = int(height * 0.5)
    scan_line = binary_mask[lookahead_y, :]
    
    # Find left and right edges of track
    white_pixels = np.where(scan_line > 0)[0]
    
    if len(white_pixels) < 2:
        return 0.0  # Can't determine, stay straight
    
    # Track edges
    left_edge = white_pixels[0]
    right_edge = white_pixels[-1]
    track_center = (left_edge + right_edge) / 2
    
    # Calculate offset from image center
    image_center = width / 2
    offset = (track_center - image_center) / (width / 2)  # Normalize to [-1, 1]
    
    return offset


def visualize_track_following(orig_img, binary_mask, offset, on_track, road_percentage):
    """
    Create visualization showing track detection and car status
    """
    # Create colored version
    height, width = orig_img.shape[:2]
    
    # Overlay binary mask on original
    overlay = orig_img.copy()
    
    # Show drivable area in green tint
    green_overlay = np.zeros_like(orig_img)
    green_overlay[:, :, 1] = binary_mask
    overlay = cv2.addWeighted(overlay, 0.7, green_overlay, 0.3, 0)
    
    # Draw center line
    image_center_x = width // 2
    cv2.line(overlay, (image_center_x, 0), (image_center_x, height), (255, 0, 255), 2)
    
    # Draw track center line
    lookahead_y = int(height * 0.5)
    scan_line = binary_mask[lookahead_y, :]
    white_pixels = np.where(scan_line > 0)[0]
    
    if len(white_pixels) >= 2:
        left_edge = white_pixels[0]
        right_edge = white_pixels[-1]
        track_center_x = int((left_edge + right_edge) / 2)
        cv2.line(overlay, (track_center_x, 0), (track_center_x, height), (0, 255, 255), 2)
        
        # Draw arrow from car to track center
        cv2.arrowedLine(overlay, (image_center_x, int(height * 0.8)),
                       (track_center_x, lookahead_y), (0, 255, 0), 3)
    
    # Show car region box
    car_y_start = int(height * 0.7)
    car_y_end = int(height * 0.9)
    car_x_start = int(width * 0.4)
    car_x_end = int(width * 0.6)
    box_color = (0, 255, 0) if on_track else (0, 0, 255)
    cv2.rectangle(overlay, (car_x_start, car_y_start), (car_x_end, car_y_end), box_color, 2)
    
    # Add text info
    status_text = "ON TRACK" if on_track else "OFF TRACK!"
    status_color = (0, 255, 0) if on_track else (0, 0, 255)
    cv2.putText(overlay, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    cv2.putText(overlay, f"Road: {road_percentage:.1%}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, f"Offset: {offset:+.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return overlay


# ============================================================
# INTERACTIVE TESTER
# ============================================================

def test_track_following():
    """
    Test the track following system interactively
    Shows binary mask and visualization
    """
    print("=" * 60)
    print("TRACK COLOR FOLLOWING - TESTER")
    print("=" * 60)
    print("\nThis will:")
    print("  1. Capture your game screen")
    print("  2. Detect road (white) vs off-track (black)")
    print("  3. Show where car should steer")
    print("  4. Detect if you go off-track")
    print("\nControls:")
    print("  'q' - Quit")
    print("  'c' - Calibrate colors (analyze current frame)")
    print("  'b' - Toggle binary view")
    print("=" * 60 + "\n")
    
    show_binary = False
    off_track_count = 0
    
    while True:
        # Capture screen
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 1024, 768)))
        
        # Process to binary
        binary_mask, _ = process_track_binary(screen)
        
        # Check if off track
        is_off_track, road_percentage = check_if_off_track(binary_mask)
        
        # Calculate steering offset
        offset = calculate_track_center(binary_mask)
        
        # Visualize
        if show_binary:
            # Show binary mask
            output = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(output, "WHITE = Road, BLACK = Off-track", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Show overlay
            output = visualize_track_following(screen, binary_mask, offset, not is_off_track, road_percentage)
        
        # Track off-track occurrences
        if is_off_track:
            off_track_count += 1
            cv2.putText(output, f"OFF TRACK! Count: {off_track_count}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow('Track Following Test', output)
        
        # Handle keys
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("\nCalibrating colors...")
            lower, upper = calibrate_road_color(screen)
        elif key == ord('b'):
            show_binary = not show_binary
            print(f"Binary view: {show_binary}")
    
    cv2.destroyAllWindows()


# ============================================================
# MAIN DEMO
# ============================================================

def simple_demo():
    """
    Simple demonstration of the concept
    Shows original vs binary side-by-side
    """
    print("=" * 60)
    print("TRACK FOLLOWING - SIMPLE DEMO")
    print("=" * 60)
    print("\nLeft:  Original game screen")
    print("Right: Binary mask (WHITE=road, BLACK=grass)")
    print("\nPress 'q' to quit")
    print("=" * 60 + "\n")
    
    while True:
        # Capture
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 1024, 768)))
        
        # Process
        binary_mask, color_info = process_track_binary(screen, show_color_info=True)
        
        # Check off-track
        is_off_track, road_pct = check_if_off_track(binary_mask)
        
        # Resize for display
        h, w = screen.shape[:2]
        screen_small = cv2.resize(screen, (w//2, h//2))
        binary_small = cv2.resize(binary_mask, (w//2, h//2))
        binary_small = cv2.cvtColor(binary_small, cv2.COLOR_GRAY2BGR)
        
        # Add status
        status = "ON TRACK" if not is_off_track else "OFF TRACK!"
        color = (0, 255, 0) if not is_off_track else (0, 0, 255)
        cv2.putText(binary_small, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Combine
        combined = np.hstack([screen_small, binary_small])
        
        cv2.imshow('Track Following Demo', combined)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'simple':
        simple_demo()
    else:
        test_track_following()
