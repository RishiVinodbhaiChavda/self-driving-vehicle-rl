"""
Quick Visual Comparison: Original vs Improved Vision
Shows the problem with edge detection and the solution
"""

import numpy as np
from PIL import ImageGrab
import cv2
from improved_vision import process_img_hybrid, process_img_color_segmentation


def original_process_img(orig_img):
    """Original method from pds.py - has the barrier problem"""
    proc_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    proc_img = cv2.Canny(proc_img, threshold1=200, threshold2=300)
    
    lines = cv2.HoughLinesP(proc_img, 1, np.pi/180, 180, 20, 15)
    
    # Draw lines
    output = cv2.cvtColor(proc_img, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            coords = line[0]
            cv2.line(output, (coords[0], coords[1]), (coords[2], coords[3]), [0, 255, 0], 3)
    
    return output


def side_by_side_comparison():
    """
    Show original vs improved side-by-side
    """
    print("=" * 60)
    print("VISION COMPARISON: Original vs Improved")
    print("=" * 60)
    print("\nLEFT:  Original (detects ALL edges, including barriers)")
    print("RIGHT: Improved (detects ONLY road edges)")
    print("\nNotice how improved version ignores barrier edges!")
    print("\nPress 'q' to quit")
    print("=" * 60 + "\n")
    
    while True:
        # Capture screen
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 800)))
        
        # Process with both methods
        original_output = original_process_img(screen)
        improved_output, _ = process_img_hybrid(screen)
        
        # Add labels
        cv2.putText(original_output, "ORIGINAL (ALL edges)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(original_output, "Problem: Detects barriers too", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.putText(improved_output, "IMPROVED (Road only)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(improved_output, "Solution: Ignores barriers!", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Combine side-by-side
        combined = np.hstack([original_output, improved_output])
        
        # Show
        cv2.imshow('Comparison: Original vs Improved', combined)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()


def problem_demonstration():
    """
    Demonstrate the specific problem with barriers
    """
    print("=" * 60)
    print("PROBLEM DEMONSTRATION")
    print("=" * 60)
    print("\nThis shows WHY the original method fails:")
    print("1. All edges detected (including barriers)")
    print("2. Color-coded: RED = bad edges, GREEN = good edges")
    print("\nPress 'q' to quit")
    print("=" * 60 + "\n")
    
    while True:
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 800)))
        
        # Get all edges (original method)
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        all_edges = cv2.Canny(gray, threshold1=200, threshold2=300)
        
        # Get road-only edges (improved method)
        _, road_edges = process_img_color_segmentation(screen)
        
        # Create colored visualization
        output = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB).copy()
        
        # Bad edges (all_edges but NOT on road) = RED
        bad_edges = cv2.bitwise_and(all_edges, cv2.bitwise_not(road_edges))
        output[:, :, 0] = np.where(bad_edges > 0, 255, output[:, :, 0])  # Red channel
        
        # Good edges (on road) = GREEN
        good_edges = cv2.bitwise_and(all_edges, road_edges)
        output[:, :, 1] = np.where(good_edges > 0, 255, output[:, :, 1])  # Green channel
        
        # Add legend
        cv2.putText(output, "RED = Bad edges (barriers, obstacles)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(output, "GREEN = Good edges (road boundaries)", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Problem: Red edges confuse the AI', output)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()


def main_menu():
    """
    Interactive menu
    """
    print("\n" + "=" * 60)
    print("VISION COMPARISON TOOL")
    print("=" * 60)
    print("\n1. Side-by-side comparison")
    print("2. Problem demonstration (color-coded)")
    print("3. Full improved vision tester (all methods)")
    print("4. Exit")
    print("\nChoice: ", end="")
    
    choice = input()
    
    if choice == '1':
        side_by_side_comparison()
    elif choice == '2':
        problem_demonstration()
    elif choice == '3':
        from improved_vision import test_vision_improvements
        test_vision_improvements()
    elif choice == '4':
        print("\nGoodbye!")
        return False
    else:
        print("\nInvalid choice, try again.")
    
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("VISION OPTIMIZATION - COMPARISON TOOL")
    print("=" * 60)
    print("\nThis tool shows:")
    print("  • The problem with original edge detection")
    print("  • How improved methods solve it")
    print("  • Side-by-side visual comparison")
    print("=" * 60)
    
    while main_menu():
        pass
