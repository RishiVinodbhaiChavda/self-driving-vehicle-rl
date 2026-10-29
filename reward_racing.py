"""
Reward System for Racing Games
Optimized for competitive kart racing like Zombie Sprint
"""

import numpy as np
import cv2
from config_zombie_sprint import REWARD_CONFIG


class RacingRewardCalculator:
    """
    Calculate rewards for racing behavior
    Focus on position, speed, and staying on track
    """
    def __init__(self):
        self.frames_survived = 0
        self.prev_position = 8  # Start assuming last place
        self.prev_on_track = True
        self.checkpoints_passed = 0
        self.laps_completed = 0
        self.total_reward_accumulated = 0
        
    def calculate_reward(self, processed_img, original_img, current_position, prev_position):
        """
        Calculate reward for racing
        
        Args:
            processed_img: Processed image with edges
            original_img: Original screen capture
            current_position: Current race position (1-8)
            prev_position: Previous race position
            
        Returns:
            reward: Float reward value
            done: Boolean whether episode should end
        """
        reward = 0.0
        done = False
        
        self.frames_survived += 1
        
        # ========================================
        # 1. POSITION REWARDS (Most important)
        # ========================================
        
        # Reward based on current position
        if current_position == 1:
            reward += REWARD_CONFIG['position_1st']
        elif current_position == 2:
            reward += REWARD_CONFIG['position_2nd']
        elif current_position == 3:
            reward += REWARD_CONFIG['position_3rd']
        
        # Reward for improving position
        if current_position < prev_position:
            reward += REWARD_CONFIG['position_improvement']
        elif current_position > prev_position:
            reward += REWARD_CONFIG['position_decline']
        
        # ========================================
        # 2. TRACK DETECTION (Staying on track)
        # ========================================
        
        on_track = self._is_on_track(processed_img, original_img)
        
        if on_track:
            reward += REWARD_CONFIG['on_track']
        else:
            reward += REWARD_CONFIG['off_track']
            # Being off track too long ends episode
            if not self.prev_on_track and not on_track:
                done = True  # Off track for 2 consecutive frames = crash
        
        self.prev_on_track = on_track
        
        # ========================================
        # 3. FORWARD MOVEMENT (Racing is about speed)
        # ========================================
        
        speed_estimate = self._estimate_speed(processed_img)
        
        if speed_estimate > 0:
            reward += REWARD_CONFIG['forward_movement']
            
            # Extra reward for high speed
            if speed_estimate > 50:
                reward += REWARD_CONFIG['high_speed']
        else:
            # Penalty for being stopped
            reward += REWARD_CONFIG['stopped']
        
        # ========================================
        # 4. SURVIVAL REWARD
        # ========================================
        
        reward += REWARD_CONFIG['survival_tick']
        
        # ========================================
        # 5. MILESTONE BONUSES
        # ========================================
        
        # Bonus for surviving longer
        if self.frames_survived % 500 == 0:
            reward += 10.0  # Stayed in race for a while
        
        # Track total reward
        self.total_reward_accumulated += reward
        
        return reward, done
    
    def _is_on_track(self, processed_img, original_img):
        """
        Determine if the kart is on the track
        Uses color analysis of the bottom-center region
        """
        height, width = original_img.shape[:2]
        
        # Sample region at bottom-center (where kart is)
        sample_y_start = int(height * 0.7)
        sample_y_end = int(height * 0.9)
        sample_x_start = int(width * 0.4)
        sample_x_end = int(width * 0.6)
        
        sample_region = original_img[sample_y_start:sample_y_end, 
                                     sample_x_start:sample_x_end]
        
        if sample_region.size == 0:
            return True  # Assume on track if can't determine
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(sample_region, cv2.COLOR_BGR2HSV)
        
        # Define track colors (gray/dark for road)
        # and grass colors (green/brown for off-road)
        
        # Track is usually gray/dark
        lower_gray = np.array([0, 0, 20])
        upper_gray = np.array([180, 50, 120])
        track_mask = cv2.inRange(hsv, lower_gray, upper_gray)
        
        # Grass is usually green or brown
        lower_grass = np.array([30, 40, 40])
        upper_grass = np.array([90, 255, 255])
        grass_mask = cv2.inRange(hsv, lower_grass, upper_grass)
        
        # Calculate percentages
        track_pixels = np.sum(track_mask > 0)
        grass_pixels = np.sum(grass_mask > 0)
        total_pixels = sample_region.shape[0] * sample_region.shape[1]
        
        track_percentage = track_pixels / total_pixels
        grass_percentage = grass_pixels / total_pixels
        
        # On track if more track than grass visible
        return track_percentage > grass_percentage
    
    def _estimate_speed(self, processed_img):
        """
        Estimate forward speed based on edge density patterns
        
        In racing games, faster movement = more motion blur = more edges
        """
        # Count edges in different vertical sections
        height = processed_img.shape[0]
        
        # Bottom third (near kart) vs top third (far away)
        bottom_section = processed_img[2*height//3:, :]
        top_section = processed_img[:height//3, :]
        
        bottom_edges = np.sum(bottom_section > 0)
        top_edges = np.sum(top_section > 0)
        
        # More edges in bottom = moving forward
        # This is a rough heuristic
        speed = max(0, bottom_edges - top_edges) / 100.0
        
        return min(speed, 100.0)  # Cap at 100
    
    def _detect_track_edges(self, processed_img):
        """
        Detect track boundaries
        Returns distance to left and right edges
        """
        height, width = processed_img.shape
        
        # Sample horizontal line at bottom third
        scan_line_y = int(height * 0.7)
        scan_line = processed_img[scan_line_y, :]
        
        # Find edges
        edges = np.where(scan_line > 0)[0]
        
        if len(edges) < 2:
            return None, None
        
        # Left edge is leftmost point, right edge is rightmost
        left_edge = edges[0]
        right_edge = edges[-1]
        
        # Calculate distance from center
        center_x = width // 2
        
        left_distance = abs(center_x - left_edge)
        right_distance = abs(right_edge - center_x)
        
        return left_distance, right_distance
    
    def detect_powerup(self, original_img):
        """
        Detect if a powerup box is nearby
        Returns True if colorful box detected
        """
        # Powerup boxes are colorful, so look for high saturation
        hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
        
        # High saturation = colorful = powerup
        saturation = hsv[:, :, 1]
        high_sat_pixels = np.sum(saturation > 150)
        
        total_pixels = original_img.shape[0] * original_img.shape[1]
        
        # If more than 2% of pixels are highly saturated, probably a powerup
        return (high_sat_pixels / total_pixels) > 0.02
    
    def reset(self):
        """Reset for new race"""
        self.frames_survived = 0
        self.prev_position = 8
        self.prev_on_track = True
        self.checkpoints_passed = 0
        self.laps_completed = 0
        self.total_reward_accumulated = 0
    
    def get_info(self):
        """Get current state information"""
        return {
            'frames_survived': self.frames_survived,
            'checkpoints_passed': self.checkpoints_passed,
            'laps_completed': self.laps_completed,
            'total_reward': self.total_reward_accumulated,
            'on_track': self.prev_on_track,
        }
