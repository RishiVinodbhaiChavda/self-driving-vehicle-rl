"""
Configuration and Optimizations for Zombie Sprint
A 3D kart racing game requiring different approach than lane-following
"""

# ============================================================
# GAME INFORMATION
# ============================================================
GAME_NAME = "Zombie Sprint"
GAME_URL = "https://www.onlinegames.io/zombie-sprint/"
GAME_TYPE = "3D Kart Racing"
GAME_PERSPECTIVE = "3rd Person / Behind Vehicle"

# ============================================================
# GAME CONTROLS (What the game expects)
# ============================================================
CONTROLS = {
    'ACCELERATE': 'W or Up Arrow',
    'BRAKE_REVERSE': 'S or Down Arrow',
    'STEER_LEFT': 'A or Left Arrow',
    'STEER_RIGHT': 'D or Right Arrow',
    'HOP_DRIFT': 'Space or Shift',
    'USE_POWERUP': 'E or Enter',
    'POWERUP_BACKWARD': 'Hold S + E',
    'LOOK_BACK': 'Q',
    'PAUSE': 'Esc or P',
}

# ============================================================
# SCREEN CAPTURE SETTINGS
# ============================================================
# Zombie Sprint requires larger capture area for 3D perspective
CAPTURE_BBOX = (0, 40, 1000, 840)  # Wider area (1000x800 instead of 800x800)
CAPTURE_WIDTH = 1000
CAPTURE_HEIGHT = 800

# ============================================================
# ACTIONS FOR RL AGENT (Optimized for Racing)
# ============================================================
# Different from lane-following - racing requires drift, powerups
ACTIONS = {
    0: {'name': 'FORWARD', 'keys': ['W'], 'description': 'Accelerate straight'},
    1: {'name': 'FORWARD_LEFT', 'keys': ['W', 'A'], 'description': 'Accelerate + steer left'},
    2: {'name': 'FORWARD_RIGHT', 'keys': ['W', 'D'], 'description': 'Accelerate + steer right'},
    3: {'name': 'DRIFT_LEFT', 'keys': ['W', 'A', 'Space'], 'description': 'Drift left around corners'},
    4: {'name': 'DRIFT_RIGHT', 'keys': ['W', 'D', 'Space'], 'description': 'Drift right around corners'},
    5: {'name': 'BRAKE', 'keys': ['S'], 'description': 'Brake/slow down'},
    6: {'name': 'USE_POWERUP', 'keys': ['W', 'E'], 'description': 'Use powerup while moving'},
    7: {'name': 'SLIGHT_LEFT', 'keys': ['W', 'A'], 'description': 'Gentle left correction'},
    8: {'name': 'SLIGHT_RIGHT', 'keys': ['W', 'D'], 'description': 'Gentle right correction'},
}

NUM_ACTIONS = len(ACTIONS)

# ============================================================
# VISION PROCESSING PARAMETERS
# ============================================================
# 3D racing needs different edge detection
VISION_CONFIG = {
    'grayscale': True,
    'canny_threshold1': 100,  # Lower threshold for 3D environments
    'canny_threshold2': 200,  # Lower than lane detection
    'blur_kernel': (7, 7),     # More blur to reduce 3D noise
    'resize_dims': (84, 84),   # Smaller for faster processing (common in DQN)
    'use_color': False,        # Could try color to detect track vs grass
    'roi_enabled': True,       # Focus on track area
    'roi_bottom': 0.8,         # Bottom 80% of screen (road)
    'roi_top': 0.3,            # Top 30% (horizon)
}

# ============================================================
# REWARD SYSTEM FOR RACING
# ============================================================
REWARD_CONFIG = {
    # Position rewards (main objective: win the race)
    'position_1st': 10.0,      # 1st place
    'position_2nd': 5.0,       # 2nd place
    'position_3rd': 3.0,       # 3rd place
    'position_improvement': 2.0,  # Moving up in rankings
    'position_decline': -2.0,     # Moving down in rankings
    
    # Speed rewards (racing is about speed)
    'forward_movement': 1.0,   # Moving forward
    'high_speed': 0.5,         # Maintaining high speed
    'stopped': -1.0,           # Being stationary
    
    # Track rewards (staying on track)
    'on_track': 0.5,           # On the racing surface
    'off_track': -5.0,         # Off-road penalty
    'wrong_direction': -3.0,   # Going backwards
    
    # Lap rewards (progress)
    'lap_completed': 20.0,     # Completed a lap
    'checkpoint_passed': 5.0,  # Passed intermediate checkpoint
    
    # Combat rewards (powerups)
    'powerup_collected': 1.0,  # Picked up powerup
    'powerup_used': 0.5,       # Used powerup effectively
    'hit_by_powerup': -2.0,    # Got hit by opponent
    
    # Crash/collision penalties
    'wall_collision': -3.0,    # Hit a wall
    'rival_collision': -1.0,   # Bumped into rival
    
    # Survival
    'survival_tick': 0.1,      # Small reward for staying in race
    
    # Race completion
    'race_finished': 50.0,     # Completed all 3 laps
    'race_failed': -10.0,      # DNF (did not finish)
}

# ============================================================
# TRAINING HYPERPARAMETERS (Optimized for Racing)
# ============================================================
TRAINING_CONFIG = {
    'num_episodes': 2000,           # More episodes for complex racing
    'max_steps_per_episode': 3000,  # Longer episodes (3 laps can take time)
    'batch_size': 64,               # Larger batch for stability
    'learning_rate': 0.0001,        # Standard DQN learning rate
    'gamma': 0.99,                  # High discount (long-term planning)
    'epsilon_start': 1.0,           # Full exploration initially
    'epsilon_end': 0.05,            # More exploration than lane-following
    'epsilon_decay': 0.995,         # Gradual decay
    'target_update_freq': 20,       # Update target network every 20 episodes
    'save_freq': 100,               # Save more frequently
    'replay_memory_size': 50000,    # Larger memory for diverse experiences
}

# ============================================================
# NETWORK ARCHITECTURE (Optimized for Racing)
# ============================================================
NETWORK_CONFIG = {
    'input_channels': 1,            # Grayscale input
    'conv_layers': [
        {'out_channels': 32, 'kernel_size': 8, 'stride': 4},
        {'out_channels': 64, 'kernel_size': 4, 'stride': 2},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1},
    ],
    'fc_layers': [512, 256],        # Two fully connected layers
    'num_actions': NUM_ACTIONS,
    'dropout': 0.3,                 # Add dropout to prevent overfitting
}

# ============================================================
# POSITION DETECTION (OCR for reading race position)
# ============================================================
POSITION_DETECTION = {
    'enabled': True,
    'method': 'template_matching',  # or 'ocr'
    'position_bbox': (50, 650, 150, 750),  # Bottom-left corner
    'update_frequency': 10,  # Check every 10 frames
}

# ============================================================
# LAP DETECTION
# ============================================================
LAP_DETECTION = {
    'enabled': True,
    'lap_bbox': (850, 50, 950, 100),  # Top-right corner
    'total_laps': 3,
}

# ============================================================
# SPEEDOMETER DETECTION
# ============================================================
SPEEDOMETER = {
    'enabled': True,
    'speed_bbox': (450, 700, 550, 780),  # Bottom center
}

# ============================================================
# RECOMMENDED CHARACTER STATS
# ============================================================
ZOMBIE_CHOICE = {
    'recommended': 'Grave Sprinter',  # Light zombie - good for learning
    'reason': 'Quick acceleration and sharp turning - easier to learn',
    'alternatives': [
        {'name': 'Neon Ghoul', 'reason': 'Similar stats, good handling'},
        {'name': 'Howler', 'reason': 'Fast acceleration'},
    ],
    'avoid_initially': [
        {'name': 'Bone Crusher', 'reason': 'Heavy - slow to turn, hard for AI'},
        {'name': 'Grave Queen', 'reason': 'Heavy - requires advanced control'},
    ]
}

# ============================================================
# RECOMMENDED TRACK FOR TRAINING
# ============================================================
TRACK_PROGRESSION = [
    {
        'name': 'Graveyard Loop',
        'difficulty': 'Easy',
        'threat_level': 'Easy',
        'episodes': '1-500',
        'reason': 'Simple layout, learn basic racing'
    },
    {
        'name': 'Graveyard Loop',
        'difficulty': 'Easy',
        'threat_level': 'Normal',
        'episodes': '501-1000',
        'reason': 'Same track, harder opponents'
    },
    {
        'name': 'Doomsday Highway',
        'difficulty': 'Normal',
        'threat_level': 'Normal',
        'episodes': '1001-1500',
        'reason': 'New track, desert environment'
    },
    {
        'name': 'Mix All Tracks',
        'difficulty': 'Various',
        'threat_level': 'Hard',
        'episodes': '1501-2000',
        'reason': 'Generalize across all tracks'
    },
]

# ============================================================
# VISUAL FEATURES TO DETECT
# ============================================================
FEATURES = {
    'track_edges': True,        # Detect track boundaries
    'rival_karts': True,        # Detect other racers
    'powerup_boxes': True,      # Detect powerup locations
    'track_curves': True,       # Anticipate turns
    'minimap': False,           # Too small, ignore initially
    'position_indicator': True, # Read race position
}

# ============================================================
# STATE REPRESENTATION
# ============================================================
STATE_CONFIG = {
    'type': 'stacked_frames',   # Stack multiple frames for motion
    'num_frames': 4,            # Stack 4 frames (common in DQN)
    'frame_size': (84, 84),     # Standard Atari size
    'normalize': True,          # Normalize to [0, 1]
}

# ============================================================
# DIFFICULTY CURVE
# ============================================================
CURRICULUM = {
    'stage_1': {
        'episodes': '1-300',
        'goal': 'Learn basic controls',
        'threat_level': 'Easy',
        'track': 'Graveyard Loop',
        'success_criteria': 'Finish race without major crashes',
    },
    'stage_2': {
        'episodes': '301-700',
        'goal': 'Learn drifting and cornering',
        'threat_level': 'Easy to Normal',
        'track': 'Graveyard Loop',
        'success_criteria': 'Finish in top 5',
    },
    'stage_3': {
        'episodes': '701-1200',
        'goal': 'Competitive racing',
        'threat_level': 'Normal',
        'track': 'Multiple tracks',
        'success_criteria': 'Finish in top 3',
    },
    'stage_4': {
        'episodes': '1201-2000',
        'goal': 'Master racing',
        'threat_level': 'Normal to Hard',
        'track': 'All tracks',
        'success_criteria': 'Win races (1st place)',
    },
}

# ============================================================
# TROUBLESHOOTING TIPS
# ============================================================
TIPS = {
    'setup': [
        '1. Open game at: https://www.onlinegames.io/zombie-sprint/',
        '2. Select "Grave Sprinter" (light zombie with good handling)',
        '3. Choose "Graveyard Loop" track',
        '4. Set threat level to "Easy"',
        '5. Position browser in TOP-LEFT corner (capture area: 1000x800)',
        '6. Start race and immediately run training script',
    ],
    'training': [
        'First 100 episodes: Agent will crash a lot - this is normal',
        'Episodes 100-500: Should start staying on track',
        'Episodes 500-1000: Should finish races regularly',
        'Episodes 1000+: Should compete for podium positions',
    ],
    'common_issues': [
        'If agent goes backwards: Increase wrong_direction penalty',
        'If agent crashes constantly: Lower speed rewards, increase on_track rewards',
        'If agent ignores powerups: Increase powerup_collected reward',
        'If training is slow: Reduce capture area or frame size',
    ]
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_action_keys(action_idx):
    """Get keyboard keys for a given action index"""
    if action_idx in ACTIONS:
        return ACTIONS[action_idx]['keys']
    return []

def get_action_name(action_idx):
    """Get action name for display"""
    if action_idx in ACTIONS:
        return ACTIONS[action_idx]['name']
    return 'UNKNOWN'

def print_config():
    """Print configuration summary"""
    print("=" * 60)
    print("ZOMBIE SPRINT - RL CONFIGURATION")
    print("=" * 60)
    print(f"\nGame: {GAME_NAME}")
    print(f"URL: {GAME_URL}")
    print(f"Type: {GAME_TYPE}")
    print(f"\nActions: {NUM_ACTIONS}")
    print(f"Capture Area: {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}")
    print(f"Frame Size: {STATE_CONFIG['frame_size']}")
    print(f"Training Episodes: {TRAINING_CONFIG['num_episodes']}")
    print(f"\nRecommended Zombie: {ZOMBIE_CHOICE['recommended']}")
    print(f"Reason: {ZOMBIE_CHOICE['reason']}")
    print(f"\nFirst Track: {TRACK_PROGRESSION[0]['name']}")
    print(f"Difficulty: {TRACK_PROGRESSION[0]['difficulty']}")
    print("=" * 60)

if __name__ == '__main__':
    print_config()
    
    print("\n" + "=" * 60)
    print("SETUP INSTRUCTIONS")
    print("=" * 60)
    for i, tip in enumerate(TIPS['setup'], 1):
        print(f"{i}. {tip}")
    
    print("\n" + "=" * 60)
    print("TRAINING EXPECTATIONS")
    print("=" * 60)
    for tip in TIPS['training']:
        print(f"• {tip}")
