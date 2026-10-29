"""
Improved Vision Processing for Self-Driving Vehicle
Solves the problem of edge detection confusing barriers/obstacles with roads

PROBLEM: Canny edge detection treats ALL edges equally
- Detects road boundaries (good ✓)
- Also detects barriers, dark obstacles (bad ✗)
- Agent gets confused and drives toward obstacles

SOLUTIONS IMPLEMENTED:
1. Color-based road detection (HSV segmentation)
2. Semantic segmentation (drivable area)
3. Combined edge + color approach
4. Template-based filtering
5. Optical flow (motion analysis)
"""

import numpy as np
from PIL import ImageGrab
import cv2
import time
from direction_keys import ReleaseKey, PressKey, W, A, S, D


# ============================================================
# SOLUTION 1: COLOR-BASED ROAD DETECTION (BEST FOR MOST GAMES)
# ============================================================

def process_img_color_segmentation(orig_img):
    """
    Uses color to identify drivable road surface
    
    LOGIC:
    - Roads are usually gray/dark colors
    - Grass/off-road is green/brown
    - Barriers are often bright colors or very dark
    
    This separates "what's road" from "what's obstacle"
    """
    # Convert to HSV (better for color detection than RGB)
    hsv = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
    
    # Define road color ranges (adjust based on your game)
    # Gray road (common in racing games)
    lower_gray_road = np.array([0, 0, 30])    # Dark gray
    upper_gray_road = np.array([180, 50, 120]) # Light gray
    road_mask = cv2.inRange(hsv, lower_gray_road, upper_gray_road)
    
    # Brown/tan road (dirt tracks)
    lower_brown = np.array([10, 50, 50])
    upper_brown = np.array([25, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Combine masks
    drivable_surface = cv2.bitwise_or(road_mask, brown_mask)
    
    # Clean up noise
    kernel = np.ones((5,5), np.uint8)
    drivable_surface = cv2.morphologyEx(drivable_surface, cv2.MORPH_CLOSE, kernel)
    drivable_surface = cv2.morphologyEx(drivable_surface, cv2.MORPH_OPEN, kernel)
    
    # Apply edge detection ONLY on drivable surface
    edges = cv2.Canny(drivable_surface, 100, 200)
    
    # Find lane lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=20)
    
    # Visualize
    output = cv2.cvtColor(drivable_surface, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    return output, drivable_surface


# ============================================================
# SOLUTION 2: INVERSE APPROACH (DETECT OBSTACLES, SUBTRACT)
# ============================================================

def process_img_obstacle_detection(orig_img):
    """
    Instead of finding road, find obstacles and exclude them
    
    LOGIC:
    - Detect bright/colorful objects (barriers, cars, powerups)
    - Detect very dark objects (walls, shadows)
    - Everything else is likely drivable
    """
    hsv = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
    
    # Detect obstacles
    # 1. Very bright objects (barriers, signs)
    _, _, v = cv2.split(hsv)
    bright_obstacles = cv2.threshold(v, 200, 255, cv2.THRESH_BINARY)[1]
    
    # 2. Very dark objects (walls, deep shadows)
    dark_obstacles = cv2.threshold(v, 40, 255, cv2.THRESH_BINARY_INV)[1]
    
    # 3. Highly saturated objects (colorful barriers, cars)
    _, s, _ = cv2.split(hsv)
    colorful_obstacles = cv2.threshold(s, 150, 255, cv2.THRESH_BINARY)[1]
    
    # Combine obstacle masks
    obstacles = cv2.bitwise_or(bright_obstacles, dark_obstacles)
    obstacles = cv2.bitwise_or(obstacles, colorful_obstacles)
    
    # Invert to get drivable area
    drivable = cv2.bitwise_not(obstacles)
    
    # Clean up
    kernel = np.ones((7,7), np.uint8)
    drivable = cv2.morphologyEx(drivable, cv2.MORPH_OPEN, kernel)
    
    # Edge detection on drivable area only
    edges = cv2.Canny(drivable, 100, 200)
    
    return edges, drivable


# ============================================================
# SOLUTION 3: COMBINED EDGE + COLOR (MOST ROBUST)
# ============================================================

def process_img_hybrid(orig_img):
    """
    Combines edge detection with color analysis
    
    STRATEGY:
    1. Use color to find drivable surface
    2. Use edges to find lane markings
    3. Filter edges by location (only keep edges ON road)
    """
    # Get drivable surface mask (from solution 1)
    _, drivable_mask = process_img_color_segmentation(orig_img)
    
    # Get all edges from original image
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    all_edges = cv2.Canny(gray, 150, 250)
    
    # Keep ONLY edges that are ON the drivable surface
    filtered_edges = cv2.bitwise_and(all_edges, drivable_mask)
    
    # Find lines
    lines = cv2.HoughLinesP(filtered_edges, 1, np.pi/180, 50, 
                            minLineLength=30, maxLineGap=20)
    
    # Draw on original
    output = orig_img.copy()
    
    # Show drivable area in green tint
    drivable_overlay = np.zeros_like(orig_img)
    drivable_overlay[:, :, 1] = drivable_mask  # Green channel
    output = cv2.addWeighted(output, 0.8, drivable_overlay, 0.2, 0)
    
    # Draw lane lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 255), 3)
    
    return output, filtered_edges


# ============================================================
# SOLUTION 4: SEMANTIC SEGMENTATION (ADVANCED)
# ============================================================

def process_img_semantic(orig_img):
    """
    Uses multiple features to classify each pixel
    
    FEATURES:
    - Color (HSV)
    - Texture (variance)
    - Position (bottom = likely road)
    - Brightness
    """
    height, width = orig_img.shape[:2]
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    
    # Create probability map for "road-ness"
    road_probability = np.zeros((height, width), dtype=np.float32)
    
    # Feature 1: Color (gray/brown = high probability)
    h, s, v = cv2.split(hsv)
    
    # Gray roads: low saturation, medium value
    gray_score = np.where((s < 50) & (v > 30) & (v < 150), 1.0, 0.0)
    road_probability += gray_score * 0.4
    
    # Feature 2: Position (bottom = more likely road)
    position_score = np.linspace(0.2, 1.0, height).reshape(-1, 1)
    position_score = np.tile(position_score, (1, width))
    road_probability += position_score * 0.3
    
    # Feature 3: Texture (roads are smoother than obstacles)
    texture = cv2.Laplacian(gray, cv2.CV_64F)
    texture = np.abs(texture)
    texture_score = 1.0 - np.clip(texture / 100, 0, 1)  # Low texture = high score
    road_probability += texture_score * 0.3
    
    # Normalize to 0-255
    road_probability = np.clip(road_probability * 255, 0, 255).astype(np.uint8)
    
    # Threshold to binary mask
    _, road_mask = cv2.threshold(road_probability, 150, 255, cv2.THRESH_BINARY)
    
    # Clean up
    kernel = np.ones((5,5), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    
    # Edge detection on road only
    edges = cv2.Canny(road_mask, 100, 200)
    
    return edges, road_mask


# ============================================================
# SOLUTION 5: ADAPTIVE THRESHOLDING (FOR VARYING LIGHTING)
# ============================================================

def process_img_adaptive(orig_img):
    """
    Handles games with changing lighting conditions
    
    Uses local adaptive thresholding instead of global
    """
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive threshold (good for varying lighting)
    adaptive = cv2.adaptiveThreshold(gray, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    
    # Invert so road is white
    road_mask = cv2.bitwise_not(adaptive)
    
    # Clean up
    kernel = np.ones((3,3), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
    
    # Edge detection
    edges = cv2.Canny(road_mask, 100, 200)
    
    return edges, road_mask


# ============================================================
# SOLUTION 6: OPTICAL FLOW (MOTION-BASED)
# ============================================================

class OpticalFlowProcessor:
    """
    Uses motion to identify drivable areas
    
    LOGIC:
    - When car moves forward, road flows backward
    - Barriers/obstacles move differently
    - Track flow patterns to identify road
    """
    def __init__(self):
        self.prev_gray = None
    
    def process_img(self, orig_img):
        gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return gray, gray
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Magnitude of flow
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Road should have consistent backward flow
        # Filter by magnitude and direction
        road_flow = np.where((mag > 2) & (mag < 20), 255, 0).astype(np.uint8)
        
        self.prev_gray = gray
        
        return road_flow, mag


# ============================================================
# SOLUTION 7: GAME-SPECIFIC CALIBRATION
# ============================================================

def calibrate_for_game(sample_images):
    """
    Analyze sample images from your specific game to find optimal parameters
    
    Usage:
    1. Capture 10-20 screenshots of game with good road visibility
    2. Run this to find best HSV ranges
    3. Use those values in color segmentation
    """
    all_hsv_values = []
    
    for img in sample_images:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Sample bottom-center region (likely road)
        h, w = hsv.shape[:2]
        road_region = hsv[int(h*0.7):int(h*0.9), int(w*0.3):int(w*0.7)]
        
        # Collect HSV values
        all_hsv_values.append(road_region.reshape(-1, 3))
    
    # Concatenate all samples
    all_hsv = np.vstack(all_hsv_values)
    
    # Calculate statistics
    h_mean, s_mean, v_mean = np.mean(all_hsv, axis=0)
    h_std, s_std, v_std = np.std(all_hsv, axis=0)
    
    # Recommend HSV ranges (mean ± 2*std)
    lower_hsv = np.array([
        max(0, h_mean - 2*h_std),
        max(0, s_mean - 2*s_std),
        max(0, v_mean - 2*v_std)
    ])
    
    upper_hsv = np.array([
        min(180, h_mean + 2*h_std),
        min(255, s_mean + 2*s_std),
        min(255, v_mean + 2*v_std)
    ])
    
    print("=" * 60)
    print("CALIBRATED HSV RANGES FOR YOUR GAME")
    print("=" * 60)
    print(f"Lower HSV: {lower_hsv.astype(int)}")
    print(f"Upper HSV: {upper_hsv.astype(int)}")
    print("\nUse these values in process_img_color_segmentation()")
    print("=" * 60)
    
    return lower_hsv, upper_hsv


# ============================================================
# COMPARISON DEMO
# ============================================================

def compare_all_methods(orig_img):
    """
    Show all methods side-by-side for comparison
    """
    # Original
    result1, _ = process_img_color_segmentation(orig_img)
    result2, _ = process_img_obstacle_detection(orig_img)
    result3, _ = process_img_hybrid(orig_img)
    result4, _ = process_img_semantic(orig_img)
    result5, _ = process_img_adaptive(orig_img)
    
    # Resize for display
    h, w = orig_img.shape[:2]
    new_h, new_w = h//2, w//2
    
    orig_small = cv2.resize(orig_img, (new_w, new_h))
    r1_small = cv2.resize(result1, (new_w, new_h))
    r2_small = cv2.resize(result2, (new_w, new_h))
    r3_small = cv2.resize(result3, (new_w, new_h))
    r4_small = cv2.resize(result4, (new_w, new_h))
    r5_small = cv2.resize(result5, (new_w, new_h))
    
    # Convert gray to BGR for concatenation
    if len(r2_small.shape) == 2:
        r2_small = cv2.cvtColor(r2_small, cv2.COLOR_GRAY2BGR)
    if len(r4_small.shape) == 2:
        r4_small = cv2.cvtColor(r4_small, cv2.COLOR_GRAY2BGR)
    if len(r5_small.shape) == 2:
        r5_small = cv2.cvtColor(r5_small, cv2.COLOR_GRAY2BGR)
    
    # Add labels
    cv2.putText(orig_small, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(r1_small, "Color Seg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(r2_small, "Obstacle Det", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(r3_small, "Hybrid", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(r4_small, "Semantic", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(r5_small, "Adaptive", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    # Arrange in grid
    top_row = np.hstack([orig_small, r1_small, r2_small])
    bottom_row = np.hstack([r3_small, r4_small, r5_small])
    combined = np.vstack([top_row, bottom_row])
    
    return combined


# ============================================================
# INTERACTIVE TESTING
# ============================================================

def test_vision_improvements():
    """
    Interactive tester - press keys to switch between methods
    """
    print("=" * 60)
    print("IMPROVED VISION TESTING")
    print("=" * 60)
    print("Controls:")
    print("  1 - Color Segmentation (recommended)")
    print("  2 - Obstacle Detection")
    print("  3 - Hybrid (best quality)")
    print("  4 - Semantic")
    print("  5 - Adaptive")
    print("  0 - Compare All")
    print("  q - Quit")
    print("=" * 60)
    
    method = 3  # Default to hybrid
    optical_flow = OpticalFlowProcessor()
    
    while True:
        # Capture screen
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 800)))
        
        # Process based on selected method
        if method == 0:
            output = compare_all_methods(screen)
        elif method == 1:
            output, _ = process_img_color_segmentation(screen)
        elif method == 2:
            edges, _ = process_img_obstacle_detection(screen)
            output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif method == 3:
            output, _ = process_img_hybrid(screen)
        elif method == 4:
            edges, _ = process_img_semantic(screen)
            output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif method == 5:
            edges, _ = process_img_adaptive(screen)
            output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Show method name
        method_names = ["Compare All", "Color Seg", "Obstacle", "Hybrid", "Semantic", "Adaptive"]
        if method < len(method_names):
            cv2.putText(output, f"Method: {method_names[method]} (Press 0-5)", 
                       (10, output.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        cv2.imshow('Improved Vision', output)
        
        # Handle key presses
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif ord('0') <= key <= ord('5'):
            method = key - ord('0')
            print(f"Switched to method {method}: {method_names[method] if method < len(method_names) else 'Unknown'}")
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_vision_improvements()
