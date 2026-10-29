"""
Reward System for Self-Driving Vehicle
Calculates rewards based on driving behavior
"""

import numpy as np
import cv2


class RewardCalculator:
    """
    Calculate rewards based on the current state of the vehicle
    Rewards encourage staying on the road and moving forward
    """
    def __init__(self):
        self.prev_center_distance = None
        self.prev_speed_estimate = 0
        self.frames_survived = 0
        
    def calculate_reward(self, processed_img, original_img):
        """
        Calculate reward based on the current frame
        
        Args:
            processed_img: Processed image with detected edges/lines
            original_img: Original screen capture
            
        Returns:
            reward: Float value representing how good the current state is
            done: Boolean indicating if episode should end
        """
        reward = 0.0
        done = False
        
        self.frames_survived += 1
        
        # 1. Check if there are detected lines (staying on road)
        line_pixels = np.sum(processed_img > 0)
        
        if line_pixels < 100:  # Very few lines detected - probably crashed or off-road
            reward = -10.0
            done = True
            return reward, done
        
        # 2. Reward for detecting good amount of lines (on the road)
        if 100 <= line_pixels < 1000:
            reward += 0.5
        elif 1000 <= line_pixels < 5000:
            reward += 1.0
        elif line_pixels >= 5000:
            reward += 0.3  # Too many lines might mean cluttered view
        
        # 3. Calculate center alignment (staying in the middle of the lane)
        center_distance = self._calculate_center_alignment(processed_img)
        
        if center_distance is not None:
            # Reward for being near center
            if center_distance < 50:
                reward += 2.0
            elif center_distance < 100:
                reward += 1.0
            elif center_distance < 200:
                reward += 0.5
            else:
                reward -= 0.5  # Penalty for being far from center
            
            # Reward for maintaining center (smooth driving)
            if self.prev_center_distance is not None:
                improvement = self.prev_center_distance - center_distance
                reward += improvement * 0.01  # Small reward for getting closer to center
            
            self.prev_center_distance = center_distance
        
        # 4. Estimate speed and reward forward movement
        speed_estimate = self._estimate_speed(processed_img)
        
        if speed_estimate > 0:
            reward += min(speed_estimate * 0.1, 1.0)  # Cap speed reward at 1.0
        
        self.prev_speed_estimate = speed_estimate
        
        # 5. Small reward for survival (encourages not crashing)
        reward += 0.1
        
        # 6. Bonus for surviving longer
        if self.frames_survived % 100 == 0:
            reward += 5.0  # Bonus every 100 frames
        
        return reward, done
    
    def _calculate_center_alignment(self, processed_img):
        """
        Calculate how far the vehicle is from the center of the detected lane
        
        Args:
            processed_img: Processed image with detected edges
            
        Returns:
            distance: Distance from center (or None if can't determine)
        """
        height, width = processed_img.shape
        
        # Focus on the bottom half of the image (closer to vehicle)
        roi_bottom = processed_img[height//2:, :]
        
        # Find left and right lane boundaries
        left_lane = None
        right_lane = None
        
        # Scan from center outward to find lanes
        center_x = width // 2
        
        # Find left boundary
        for x in range(center_x, 0, -1):
            if np.sum(roi_bottom[:, x]) > 50:  # Found lane marking
                left_lane = x
                break
        
        # Find right boundary
        for x in range(center_x, width):
            if np.sum(roi_bottom[:, x]) > 50:  # Found lane marking
                right_lane = x
                break
        
        # Calculate center of lane
        if left_lane is not None and right_lane is not None:
            lane_center = (left_lane + right_lane) // 2
            distance = abs(center_x - lane_center)
            return distance
        
        return None
    
    def _estimate_speed(self, processed_img):
        """
        Estimate speed based on the amount of motion blur or line density
        This is a rough estimate - in a real game, you'd read the speedometer
        
        Args:
            processed_img: Processed image
            
        Returns:
            speed_estimate: Rough estimate of forward speed
        """
        # Count lines in bottom third (near vehicle) vs top third (far away)
        height = processed_img.shape[0]
        
        bottom_third = processed_img[2*height//3:, :]
        top_third = processed_img[:height//3, :]
        
        bottom_lines = np.sum(bottom_third > 0)
        top_lines = np.sum(top_third > 0)
        
        # If more lines in bottom than top, we're likely moving forward
        if bottom_lines > top_lines:
            return min((bottom_lines - top_lines) / 1000.0, 5.0)
        
        return 0.0
    
    def reset(self):
        """Reset the reward calculator for a new episode"""
        self.prev_center_distance = None
        self.prev_speed_estimate = 0
        self.frames_survived = 0
    
    def get_info(self):
        """Get additional information about the current state"""
        return {
            'frames_survived': self.frames_survived,
            'prev_center_distance': self.prev_center_distance,
            'prev_speed_estimate': self.prev_speed_estimate
        }
