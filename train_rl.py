"""
Training Script for Self-Driving Vehicle RL Agent
Combines vision processing, reward calculation, and DQN learning
"""

import numpy as np
from PIL import ImageGrab
import cv2
import time
from direction_keys import ReleaseKey, PressKey, W, A, S, D
from rl_model import DQNAgent
from reward_system import RewardCalculator
import torch


class DrivingEnvironment:
    """
    Environment wrapper for the driving game
    Handles screen capture, processing, action execution, and reward calculation
    """
    def __init__(self):
        self.reward_calculator = RewardCalculator()
        self.current_keys = []
        
    def process_img(self, orig_img):
        """
        Process the original image for both display and RL input
        
        Args:
            orig_img: Original screen capture
            
        Returns:
            display_img: Image to display (with lines drawn)
            state_img: Processed image for RL agent (normalized)
        """
        # Convert to grayscale
        proc_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        proc_img = cv2.Canny(proc_img, threshold1=200, threshold2=300)
        
        # Apply Hough Line Transform
        lines = cv2.HoughLinesP(proc_img, 1, np.pi/180, 180, 20, 15)
        
        # Create a copy for display
        display_img = proc_img.copy()
        
        # Draw lines on display image
        if lines is not None:
            for line in lines:
                coords = line[0]
                cv2.line(display_img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 255, 255], 3)
        
        # Prepare state for RL (resize and normalize)
        state_img = cv2.resize(proc_img, (760, 760))
        state_img = state_img.astype(np.float32) / 255.0
        state_img = np.expand_dims(state_img, axis=0)  # Add channel dimension
        
        return display_img, state_img, proc_img
    
    def execute_action(self, action_idx):
        """
        Execute the action selected by the agent
        
        Args:
            action_idx: Index of the action to execute
        """
        # Release all previous keys
        for key in self.current_keys:
            ReleaseKey(key)
        self.current_keys = []
        
        # Execute new action
        if action_idx == 0:  # W (Forward)
            PressKey(W)
            self.current_keys.append(W)
        elif action_idx == 1:  # A (Left)
            PressKey(A)
            self.current_keys.append(A)
        elif action_idx == 2:  # S (Backward)
            PressKey(S)
            self.current_keys.append(S)
        elif action_idx == 3:  # D (Right)
            PressKey(D)
            self.current_keys.append(D)
        elif action_idx == 4:  # WA (Forward + Left)
            PressKey(W)
            PressKey(A)
            self.current_keys.extend([W, A])
        elif action_idx == 5:  # WD (Forward + Right)
            PressKey(W)
            PressKey(D)
            self.current_keys.extend([W, D])
        # action_idx == 6 is NONE (no action)
    
    def reset(self):
        """Reset the environment for a new episode"""
        # Release all keys
        for key in self.current_keys:
            ReleaseKey(key)
        self.current_keys = []
        
        # Reset reward calculator
        self.reward_calculator.reset()
        
        # Wait a moment
        time.sleep(0.5)
        
        # Get initial state
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 800)))
        _, state_img, _ = self.process_img(screen)
        
        return state_img
    
    def step(self, action_idx):
        """
        Execute one step in the environment
        
        Args:
            action_idx: Action to execute
            
        Returns:
            next_state: Next state after action
            reward: Reward for the transition
            done: Whether episode is done
            info: Additional information
        """
        # Execute action
        self.execute_action(action_idx)
        
        # Wait for action to take effect
        time.sleep(0.1)
        
        # Capture new screen
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 800)))
        
        # Process image
        display_img, next_state, proc_img = self.process_img(screen)
        
        # Calculate reward
        reward, done = self.reward_calculator.calculate_reward(proc_img, screen)
        
        # Get additional info
        info = self.reward_calculator.get_info()
        
        return next_state, reward, done, info, display_img
    
    def cleanup(self):
        """Release all keys when done"""
        for key in self.current_keys:
            ReleaseKey(key)
        self.current_keys = []


def train_agent(num_episodes=1000, max_steps_per_episode=1000, 
                batch_size=32, target_update_freq=10, save_freq=50):
    """
    Train the RL agent
    
    Args:
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        batch_size: Batch size for training
        target_update_freq: How often to update target network
        save_freq: How often to save the model
    """
    print("=" * 60)
    print("SELF-DRIVING VEHICLE - REINFORCEMENT LEARNING TRAINING")
    print("=" * 60)
    print("\nInitializing agent and environment...")
    
    # Create agent and environment
    agent = DQNAgent()
    env = DrivingEnvironment()
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    print("\nTraining setup:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Max steps per episode: {max_steps_per_episode}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {agent.device}")
    print("\n" + "=" * 60)
    print("IMPORTANT: Start your driving game now!")
    print("Position the game window in the TOP-LEFT corner of your screen")
    print("Training will begin in 5 seconds...")
    print("=" * 60 + "\n")
    
    time.sleep(5)
    
    try:
        for episode in range(num_episodes):
            # Reset environment
            state = env.reset()
            episode_reward = 0
            episode_loss = []
            
            print(f"\n[Episode {episode + 1}/{num_episodes}] Starting...")
            
            for step in range(max_steps_per_episode):
                # Select action
                action = agent.select_action(state)
                
                # Execute action
                next_state, reward, done, info, display_img = env.step(action)
                
                # Store transition
                agent.store_transition(state, action, reward, next_state, done)
                
                # Train the agent
                loss = agent.train_step(batch_size)
                if loss is not None:
                    episode_loss.append(loss)
                
                # Update state
                state = next_state
                episode_reward += reward
                
                # Add info text directly on the image
                info_text = f"Ep: {episode+1} | Step: {step+1} | Reward: {episode_reward:.2f}"
                epsilon_text = f"Epsilon: {agent.epsilon:.3f}"
                cv2.putText(display_img, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_img, epsilon_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display the processed image
                cv2.imshow('Self-Driving Vehicle - Training', display_img)
                
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
            
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            losses.append(avg_loss)
            
            # Decay exploration rate
            agent.decay_epsilon()
            
            # Update target network
            if (episode + 1) % target_update_freq == 0:
                agent.update_target_network()
                print(f"  → Target network updated")
            
            # Print episode summary
            avg_reward_last_10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            print(f"[Episode {episode + 1}] Finished!")
            print(f"  Steps: {step + 1} | Total Reward: {episode_reward:.2f}")
            print(f"  Avg Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.3f}")
            print(f"  Avg Reward (last 10): {avg_reward_last_10:.2f}")
            print(f"  Frames Survived: {info['frames_survived']}")
            
            # Save model periodically
            if (episode + 1) % save_freq == 0:
                agent.save_model(f'models/dqn_model_episode_{episode + 1}.pth')
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
        agent.save_model('models/dqn_model_final.pth')
        print("\nFinal model saved!")
        
        # Print training summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Total Episodes: {len(episode_rewards)}")
        if len(episode_rewards) > 0:
            print(f"Average Reward: {np.mean(episode_rewards):.2f}")
            print(f"Max Reward: {np.max(episode_rewards):.2f}")
            print(f"Average Episode Length: {np.mean(episode_lengths):.2f}")
        else:
            print("No episodes completed")
        print(f"Final Epsilon: {agent.epsilon:.3f}")


if __name__ == '__main__':
    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    
    # Start training
    train_agent(num_episodes=1000, max_steps_per_episode=1000)
