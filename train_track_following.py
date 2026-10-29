"""
Training Script for Track Color Following
Uses binary mask approach: WHITE=road, BLACK=off-track
Automatically presses 'R' and gives penalty when off-track
"""

import numpy as np
from PIL import ImageGrab
import cv2
import time
from direction_keys import ReleaseKey, PressKey, W, A, S, D
from track_color_following import (
    process_track_binary, 
    check_if_off_track, 
    calculate_track_center,
    press_r_key,
    visualize_track_following
)
from rl_model import DQNAgent
import torch


class TrackFollowingEnvironment:
    """
    Environment for track-following with automatic reset on off-track
    """
    def __init__(self, off_track_threshold=0.3, reset_on_off_track=True):
        self.off_track_threshold = off_track_threshold
        self.reset_on_off_track = reset_on_off_track
        self.current_keys = []
        self.off_track_count = 0
        self.total_frames = 0
        self.reset_count = 0
        
    def process_img(self, orig_img):
        """
        Process image to binary track mask
        
        Returns:
            display_img: Visualization
            state_img: Binary mask for RL (760x760)
            binary_mask: Full size binary mask
            is_off_track: Boolean
            track_offset: Float [-1, 1]
        """
        # Get binary mask
        binary_mask, _ = process_track_binary(orig_img)
        
        # Check if off track
        is_off_track, road_percentage = check_if_off_track(
            binary_mask, self.off_track_threshold
        )
        
        # Calculate track center offset
        track_offset = calculate_track_center(binary_mask)
        
        # Create visualization
        display_img = visualize_track_following(
            orig_img, binary_mask, track_offset, 
            not is_off_track, road_percentage
        )
        
        # Prepare state for RL
        state_img = cv2.resize(binary_mask, (760, 760))
        state_img = state_img.astype(np.float32) / 255.0
        state_img = np.expand_dims(state_img, axis=0)
        
        return display_img, state_img, binary_mask, is_off_track, track_offset
    
    def execute_action(self, action_idx):
        """Execute driving action"""
        # Release all keys
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
        # 6 = no action
    
    def reset_game(self):
        """Press R to reset game and wait"""
        print("  → Pressing 'R' to reset game...")
        
        # Release all keys first
        for key in self.current_keys:
            ReleaseKey(key)
        self.current_keys = []
        
        # Press R
        press_r_key()
        
        # Wait for game to reset
        time.sleep(2.0)
        
        self.reset_count += 1
    
    def reset(self):
        """Reset environment for new episode"""
        # Release all keys
        for key in self.current_keys:
            ReleaseKey(key)
        self.current_keys = []
        
        # Reset counters
        self.off_track_count = 0
        self.total_frames = 0
        
        time.sleep(0.5)
        
        # Get initial state
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 1024, 768)))
        _, state_img, _, _, _ = self.process_img(screen)
        
        return state_img
    
    def step(self, action_idx):
        """
        Execute one step
        
        Returns:
            next_state, reward, done, info, display_img
        """
        # Execute action
        self.execute_action(action_idx)
        
        # Wait for action effect
        time.sleep(0.05)
        
        # Capture screen
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 1024, 768)))
        
        # Process
        display_img, next_state, binary_mask, is_off_track, track_offset = self.process_img(screen)
        
        self.total_frames += 1
        
        # Calculate reward
        reward = 0.0
        done = False
        
        if is_off_track:
            self.off_track_count += 1
            
            # PENALTY for being off track
            reward = -10.0
            
            # If off track for too long, reset
            if self.off_track_count >= 5:  # Off track for 5 consecutive frames
                if self.reset_on_off_track:
                    self.reset_game()
                done = True
                reward = -20.0  # Extra penalty for causing reset
        else:
            # REWARDS for staying on track
            self.off_track_count = 0  # Reset counter
            
            # Base reward for being on track
            reward += 1.0
            
            # Reward for staying centered
            center_reward = 2.0 * (1.0 - abs(track_offset))
            reward += center_reward
            
            # Reward for moving forward (assume moving if on track)
            reward += 0.5
            
            # Small survival bonus
            reward += 0.1
        
        # Bonus for staying on track for a while
        if self.total_frames % 100 == 0 and self.off_track_count == 0:
            reward += 5.0
        
        info = {
            'off_track_count': self.off_track_count,
            'total_frames': self.total_frames,
            'track_offset': track_offset,
            'reset_count': self.reset_count
        }
        
        return next_state, reward, done, info, display_img
    
    def cleanup(self):
        """Release all keys"""
        for key in self.current_keys:
            ReleaseKey(key)
        self.current_keys = []


def train_agent(num_episodes=1000, max_steps_per_episode=1000,
                batch_size=32, target_update_freq=10, save_freq=50):
    """
    Train agent with track color following
    """
    print("=" * 60)
    print("TRACK COLOR FOLLOWING - RL TRAINING")
    print("=" * 60)
    print("\nInitializing agent and environment...")
    
    # Create agent and environment
    agent = DQNAgent(num_actions=7)  # W, A, S, D, WA, WD, None
    env = TrackFollowingEnvironment(
        off_track_threshold=0.3,
        reset_on_off_track=True
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    off_track_episodes = []
    losses = []
    
    print("\nTraining configuration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Max steps per episode: {max_steps_per_episode}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {agent.device}")
    print(f"  Auto-reset on off-track: Enabled")
    
    print("\n" + "=" * 60)
    print("GAME SETUP:")
    print("=" * 60)
    print("1. Start your driving game")
    print("2. Position window in TOP-LEFT corner")
    print("3. Make sure game is running (car on track)")
    print("4. Training will begin in 5 seconds...")
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
                
                # Train
                loss = agent.train_step(batch_size)
                if loss is not None:
                    episode_loss.append(loss)
                
                # Update state
                state = next_state
                episode_reward += reward
                
                # Add info to display
                cv2.putText(display_img, f"Ep: {episode+1} | Step: {step+1}", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display_img, f"Reward: {episode_reward:.1f}", (10, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display_img, f"Epsilon: {agent.epsilon:.3f}", (10, 210),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display_img, f"Resets: {info['reset_count']}", (10, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Display
                cv2.imshow('Track Following - Training', display_img)
                
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
            off_track_episodes.append(info['off_track_count'])
            
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            losses.append(avg_loss)
            
            # Decay epsilon
            agent.decay_epsilon()
            
            # Update target network
            if (episode + 1) % target_update_freq == 0:
                agent.update_target_network()
                print(f"  → Target network updated")
            
            # Print episode summary
            avg_reward_last_10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            avg_off_track = np.mean(off_track_episodes[-10:]) if len(off_track_episodes) >= 10 else np.mean(off_track_episodes)
            
            print(f"[Episode {episode + 1}] Finished!")
            print(f"  Steps: {step + 1} | Total Reward: {episode_reward:.2f}")
            print(f"  Avg Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.3f}")
            print(f"  Avg Reward (last 10): {avg_reward_last_10:.2f}")
            print(f"  Avg Off-track (last 10): {avg_off_track:.1f}")
            print(f"  Total Resets: {info['reset_count']}")
            
            # Save model periodically
            if (episode + 1) % save_freq == 0:
                agent.save_model(f'models/track_following_episode_{episode + 1}.pth')
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
        agent.save_model('models/track_following_final.pth')
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
            print(f"Average Off-track Count: {np.mean(off_track_episodes):.1f}")
            print(f"Total Resets: {env.reset_count}")
            print(f"Final Epsilon: {agent.epsilon:.3f}")


if __name__ == '__main__':
    # Create models directory
    import os
    os.makedirs('models', exist_ok=True)
    
    # Start training
    train_agent(num_episodes=1000, max_steps_per_episode=1000)
