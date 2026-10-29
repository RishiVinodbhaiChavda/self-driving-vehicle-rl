"""
Training Script for Zombie Sprint Racing Game
Optimized for 3D kart racing with drifting and powerups
"""

import numpy as np
from PIL import ImageGrab
import cv2
import time
from direction_keys import ReleaseKey, PressKey, W, A, S, D
from rl_model_racing import DQNRacingAgent
from reward_racing import RacingRewardCalculator
import torch
from config_zombie_sprint import *
from collections import deque


# Extended key codes for Space and E
try:
    import ctypes
    SendInput = ctypes.windll.user32.SendInput
    SPACE = 0x39
    E_KEY = 0x12
except:
    SPACE = None
    E_KEY = None
    print("Warning: Extended keys (Space, E) may not work")


class ZombieSprintEnvironment:
    """
    Environment wrapper for Zombie Sprint racing game
    """
    def __init__(self):
        self.reward_calculator = RacingRewardCalculator()
        self.current_keys = []
        self.frame_buffer = deque(maxlen=STATE_CONFIG['num_frames'])
        self.position_history = deque(maxlen=30)  # Track position changes
        self.lap_count = 0
        self.prev_position = 8  # Start assuming last place
        
    def press_key(self, hex_code):
        """Press a key using hex code"""
        extra = ctypes.c_ulong(0)
        ii_ = ctypes.c_ulong(0)
        
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
        
        ii_.ki = KeyBdInput(0, hex_code, 0x0008, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), Input_I(ki=ii_.ki))
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    
    def release_key(self, hex_code):
        """Release a key using hex code"""
        extra = ctypes.c_ulong(0)
        ii_ = ctypes.c_ulong(0)
        
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
        
        ii_.ki = KeyBdInput(0, hex_code, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), Input_I(ki=ii_.ki))
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    
    def process_img(self, orig_img):
        """
        Process image for racing game
        Focus on track detection and edges
        """
        # Resize to working size first
        orig_img = cv2.resize(orig_img, (CAPTURE_WIDTH, CAPTURE_HEIGHT))
        
        # Convert to grayscale
        gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise (3D games are noisy)
        gray = cv2.GaussianBlur(gray, VISION_CONFIG['blur_kernel'], 0)
        
        # Apply edge detection with adjusted thresholds for 3D
        edges = cv2.Canny(gray, 
                         VISION_CONFIG['canny_threshold1'], 
                         VISION_CONFIG['canny_threshold2'])
        
        # Apply ROI - focus on track area (bottom portion)
        if VISION_CONFIG['roi_enabled']:
            height = edges.shape[0]
            roi_mask = np.zeros_like(edges)
            roi_top = int(height * VISION_CONFIG['roi_top'])
            roi_bottom = int(height * VISION_CONFIG['roi_bottom'])
            roi_mask[roi_top:roi_bottom, :] = 255
            edges = cv2.bitwise_and(edges, roi_mask)
        
        # Create display image
        display_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Resize to network input size
        state_img = cv2.resize(edges, VISION_CONFIG['resize_dims'])
        state_img = state_img.astype(np.float32) / 255.0
        
        return display_img, state_img, edges
    
    def get_stacked_state(self):
        """Get stacked frames as state"""
        if len(self.frame_buffer) < STATE_CONFIG['num_frames']:
            # Pad with zeros if not enough frames
            while len(self.frame_buffer) < STATE_CONFIG['num_frames']:
                self.frame_buffer.append(np.zeros(VISION_CONFIG['resize_dims']))
        
        # Stack frames: (num_frames, height, width)
        stacked = np.stack(list(self.frame_buffer), axis=0)
        return stacked
    
    def execute_action(self, action_idx):
        """
        Execute racing action
        Includes drifting and powerup usage
        """
        # Release all previous keys
        for key_code in self.current_keys:
            if key_code == SPACE:
                self.release_key(SPACE)
            elif key_code == E_KEY:
                self.release_key(E_KEY)
            else:
                ReleaseKey(key_code)
        self.current_keys = []
        
        # Get action keys
        action_keys = get_action_keys(action_idx)
        
        # Execute new action
        for key_name in action_keys:
            if key_name == 'W':
                PressKey(W)
                self.current_keys.append(W)
            elif key_name == 'A':
                PressKey(A)
                self.current_keys.append(A)
            elif key_name == 'S':
                PressKey(S)
                self.current_keys.append(S)
            elif key_name == 'D':
                PressKey(D)
                self.current_keys.append(D)
            elif key_name == 'Space' and SPACE:
                self.press_key(SPACE)
                self.current_keys.append(SPACE)
            elif key_name == 'E' and E_KEY:
                self.press_key(E_KEY)
                self.current_keys.append(E_KEY)
    
    def detect_position(self, screen):
        """
        Attempt to detect race position from screen
        Returns estimated position (1-8)
        """
        # This is a placeholder - in real implementation, would use OCR
        # For now, return random variation around previous position
        if np.random.random() < 0.1:  # 10% chance of position change
            self.prev_position = max(1, min(8, self.prev_position + np.random.choice([-1, 0, 1])))
        return self.prev_position
    
    def reset(self):
        """Reset environment for new race"""
        # Release all keys
        for key_code in self.current_keys:
            if key_code == SPACE:
                self.release_key(SPACE)
            elif key_code == E_KEY:
                self.release_key(E_KEY)
            else:
                ReleaseKey(key_code)
        self.current_keys = []
        
        # Reset reward calculator
        self.reward_calculator.reset()
        
        # Clear frame buffer
        self.frame_buffer.clear()
        
        # Reset tracking variables
        self.position_history.clear()
        self.lap_count = 0
        self.prev_position = 8
        
        # Wait a moment
        time.sleep(0.5)
        
        # Get initial state
        screen = np.array(ImageGrab.grab(bbox=CAPTURE_BBOX))
        _, state_img, _ = self.process_img(screen)
        self.frame_buffer.append(state_img)
        
        return self.get_stacked_state()
    
    def step(self, action_idx):
        """
        Execute one step in the racing environment
        """
        # Execute action
        self.execute_action(action_idx)
        
        # Wait for action to take effect (shorter for racing responsiveness)
        time.sleep(0.05)
        
        # Capture new screen
        screen = np.array(ImageGrab.grab(bbox=CAPTURE_BBOX))
        
        # Process image
        display_img, state_img, proc_img = self.process_img(screen)
        self.frame_buffer.append(state_img)
        
        # Detect current position
        current_position = self.detect_position(screen)
        self.position_history.append(current_position)
        
        # Calculate reward
        reward, done = self.reward_calculator.calculate_reward(
            proc_img, screen, current_position, self.prev_position
        )
        
        # Update previous position
        self.prev_position = current_position
        
        # Get stacked state
        next_state = self.get_stacked_state()
        
        # Get additional info
        info = self.reward_calculator.get_info()
        info['position'] = current_position
        info['lap'] = self.lap_count
        
        return next_state, reward, done, info, display_img
    
    def cleanup(self):
        """Release all keys when done"""
        for key_code in self.current_keys:
            if key_code == SPACE:
                self.release_key(SPACE)
            elif key_code == E_KEY:
                self.release_key(E_KEY)
            else:
                ReleaseKey(key_code)
        self.current_keys = []


def train_agent():
    """
    Train the RL agent for Zombie Sprint
    """
    print("=" * 60)
    print("ZOMBIE SPRINT - REINFORCEMENT LEARNING TRAINING")
    print("=" * 60)
    print("\nInitializing racing agent and environment...")
    
    # Create agent and environment
    agent = DQNRacingAgent(
        num_actions=NUM_ACTIONS,
        learning_rate=TRAINING_CONFIG['learning_rate'],
        gamma=TRAINING_CONFIG['gamma'],
        epsilon_start=TRAINING_CONFIG['epsilon_start'],
        epsilon_end=TRAINING_CONFIG['epsilon_end'],
        epsilon_decay=TRAINING_CONFIG['epsilon_decay']
    )
    env = ZombieSprintEnvironment()
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    episode_positions = []
    losses = []
    
    print("\nTraining configuration:")
    print(f"  Episodes: {TRAINING_CONFIG['num_episodes']}")
    print(f"  Max steps per episode: {TRAINING_CONFIG['max_steps_per_episode']}")
    print(f"  Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"  Actions: {NUM_ACTIONS} (includes drifting and powerups)")
    print(f"  Device: {agent.device}")
    print(f"  Frame stacking: {STATE_CONFIG['num_frames']} frames")
    
    print("\n" + "=" * 60)
    print("GAME SETUP INSTRUCTIONS:")
    print("=" * 60)
    for tip in TIPS['setup']:
        print(f"  {tip}")
    print("\nTraining will begin in 10 seconds...")
    print("Press Ctrl+C to stop training anytime")
    print("=" * 60 + "\n")
    
    time.sleep(10)
    
    try:
        for episode in range(TRAINING_CONFIG['num_episodes']):
            # Reset environment
            state = env.reset()
            episode_reward = 0
            episode_loss = []
            best_position = 8
            
            print(f"\n[Episode {episode + 1}/{TRAINING_CONFIG['num_episodes']}] Starting race...")
            
            for step in range(TRAINING_CONFIG['max_steps_per_episode']):
                # Select action
                action = agent.select_action(state)
                action_name = get_action_name(action)
                
                # Execute action
                next_state, reward, done, info, display_img = env.step(action)
                
                # Store transition
                agent.store_transition(state, action, reward, next_state, done)
                
                # Train the agent
                loss = agent.train_step(TRAINING_CONFIG['batch_size'])
                if loss is not None:
                    episode_loss.append(loss)
                
                # Update state
                state = next_state
                episode_reward += reward
                
                # Track best position
                if info['position'] < best_position:
                    best_position = info['position']
                
                # Add info to display
                cv2.putText(display_img, f"Ep: {episode+1} | Step: {step+1}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_img, f"Action: {action_name}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_img, f"Position: {info['position']}/8", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display_img, f"Reward: {episode_reward:.1f}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display_img, f"Epsilon: {agent.epsilon:.3f}", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                # Display
                cv2.imshow('Zombie Sprint - RL Training', display_img)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n\nTraining interrupted by user!")
                    env.cleanup()
                    cv2.destroyAllWindows()
                    return
                
                if done:
                    break
            
            # Episode finished
            episode_rewards.append(episode_reward)
            episode_lengths.append(step + 1)
            episode_positions.append(best_position)
            
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            losses.append(avg_loss)
            
            # Decay exploration rate
            agent.decay_epsilon()
            
            # Update target network
            if (episode + 1) % TRAINING_CONFIG['target_update_freq'] == 0:
                agent.update_target_network()
                print(f"  → Target network updated")
            
            # Print episode summary
            avg_reward_last_10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            avg_position_last_10 = np.mean(episode_positions[-10:]) if len(episode_positions) >= 10 else np.mean(episode_positions)
            
            print(f"[Episode {episode + 1}] Race finished!")
            print(f"  Steps: {step + 1} | Total Reward: {episode_reward:.2f}")
            print(f"  Best Position: {best_position}/8 | Avg Loss: {avg_loss:.4f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Avg Reward (last 10): {avg_reward_last_10:.2f}")
            print(f"  Avg Position (last 10): {avg_position_last_10:.1f}/8")
            
            # Save model periodically
            if (episode + 1) % TRAINING_CONFIG['save_freq'] == 0:
                agent.save_model(f'models/zombie_sprint_episode_{episode + 1}.pth')
                print(f"  → Model saved!")
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted!")
    
    finally:
        # Cleanup
        env.cleanup()
        cv2.destroyAllWindows()
        
        # Save final model
        agent.save_model('models/zombie_sprint_final.pth')
        print("\nFinal model saved!")
        
        # Print training summary
        if len(episode_rewards) > 0:
            print("\n" + "=" * 60)
            print("TRAINING SUMMARY")
            print("=" * 60)
            print(f"Total Episodes: {len(episode_rewards)}")
            print(f"Average Reward: {np.mean(episode_rewards):.2f}")
            print(f"Max Reward: {np.max(episode_rewards):.2f}")
            print(f"Average Episode Length: {np.mean(episode_lengths):.2f}")
            print(f"Average Best Position: {np.mean(episode_positions):.1f}/8")
            print(f"Best Ever Position: {min(episode_positions)}/8")
            print(f"Final Epsilon: {agent.epsilon:.3f}")


if __name__ == '__main__':
    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    
    # Print configuration
    print_config()
    print()
    
    # Start training
    train_agent()
