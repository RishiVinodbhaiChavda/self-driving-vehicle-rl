"""
Test Script for Trained RL Agent
Load a trained model and watch it drive
"""

import numpy as np
from PIL import ImageGrab
import cv2
import time
from direction_keys import ReleaseKey, PressKey, W, A, S, D
from rl_model import DQNAgent
import torch


class DrivingEnvironment:
    """Environment for testing the trained agent"""
    def __init__(self):
        self.current_keys = []
        
    def process_img(self, orig_img):
        """Process image for agent input and display"""
        proc_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        proc_img = cv2.Canny(proc_img, threshold1=200, threshold2=300)
        
        lines = cv2.HoughLinesP(proc_img, 1, np.pi/180, 180, 20, 15)
        
        display_img = proc_img.copy()
        
        if lines is not None:
            for line in lines:
                coords = line[0]
                cv2.line(display_img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 255, 255], 3)
        
        # Prepare state for RL
        state_img = cv2.resize(proc_img, (760, 760))
        state_img = state_img.astype(np.float32) / 255.0
        state_img = np.expand_dims(state_img, axis=0)
        
        return display_img, state_img
    
    def execute_action(self, action_idx, action_name):
        """Execute action and return action name"""
        # Release all previous keys
        for key in self.current_keys:
            ReleaseKey(key)
        self.current_keys = []
        
        # Execute new action
        if action_idx == 0:  # W
            PressKey(W)
            self.current_keys.append(W)
        elif action_idx == 1:  # A
            PressKey(A)
            self.current_keys.append(A)
        elif action_idx == 2:  # S
            PressKey(S)
            self.current_keys.append(S)
        elif action_idx == 3:  # D
            PressKey(D)
            self.current_keys.append(D)
        elif action_idx == 4:  # WA
            PressKey(W)
            PressKey(A)
            self.current_keys.extend([W, A])
        elif action_idx == 5:  # WD
            PressKey(W)
            PressKey(D)
            self.current_keys.extend([W, D])
        
        return action_name
    
    def cleanup(self):
        """Release all keys"""
        for key in self.current_keys:
            ReleaseKey(key)
        self.current_keys = []


def test_agent(model_path, duration=300):
    """
    Test a trained agent
    
    Args:
        model_path: Path to the saved model
        duration: How long to run in seconds
    """
    print("=" * 60)
    print("SELF-DRIVING VEHICLE - TESTING MODE")
    print("=" * 60)
    print(f"\nLoading model from: {model_path}")
    
    # Create agent and load model
    agent = DQNAgent()
    agent.load_model(model_path)
    agent.epsilon = 0.0  # No exploration during testing
    
    # Create environment
    env = DrivingEnvironment()
    
    # Action names
    action_names = {
        0: 'FORWARD (W)',
        1: 'LEFT (A)',
        2: 'BACK (S)',
        3: 'RIGHT (D)',
        4: 'FWD+LEFT (WA)',
        5: 'FWD+RIGHT (WD)',
        6: 'NO ACTION'
    }
    
    print("\n" + "=" * 60)
    print("IMPORTANT: Start your driving game now!")
    print("Position the game window in the TOP-LEFT corner of your screen")
    print("Testing will begin in 5 seconds...")
    print("Press 'q' to quit")
    print("=" * 60 + "\n")
    
    time.sleep(5)
    
    start_time = time.time()
    frame_count = 0
    
    try:
        print("Testing started!\n")
        
        while True:
            # Check duration
            elapsed = time.time() - start_time
            if elapsed > duration:
                break
            
            # Capture screen
            screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 800)))
            
            # Process image
            display_img, state = env.process_img(screen)
            
            # Select action (greedy - no exploration)
            action_idx = agent.select_action(state)
            action_name = action_names[action_idx]
            
            # Execute action
            env.execute_action(action_idx, action_name)
            
            # Add action text to display
            cv2.putText(display_img, f"Action: {action_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_img, f"Time: {int(elapsed)}s", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_img, f"Frame: {frame_count}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display
            cv2.imshow('Self-Driving Vehicle - Testing', display_img)
            
            frame_count += 1
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nTesting stopped by user!")
                break
            
            time.sleep(0.05)  # Small delay between actions
        
        print("\n" + "=" * 60)
        print("TESTING COMPLETED!")
        print("=" * 60)
        print(f"Total time: {elapsed:.2f}s")
        print(f"Total frames: {frame_count}")
        print(f"FPS: {frame_count/elapsed:.2f}")
        
    except KeyboardInterrupt:
        print("\nTesting interrupted!")
    
    finally:
        env.cleanup()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Default to final model
        model_path = 'models/dqn_model_final.pth'
    
    test_agent(model_path, duration=300)
