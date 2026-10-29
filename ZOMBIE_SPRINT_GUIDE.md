# 🏁 Zombie Sprint - RL Training Guide

Complete guide for training an AI agent to race in Zombie Sprint using Deep Reinforcement Learning.

---

## 🎮 Game Overview

**Zombie Sprint** is a 3D kart racing game where eight zombie racers compete across four tracks. Unlike simple lane-following games, this requires:

- **Strategic racing**: Overtaking opponents, blocking, defending position
- **Drifting**: Corner technique to maintain speed through turns
- **Power-ups**: Collecting and using items strategically
- **Track learning**: Memorizing each track's layout and optimal racing lines

**Game URL**: [https://www.onlinegames.io/zombie-sprint/](https://www.onlinegames.io/zombie-sprint/)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision opencv-python numpy Pillow

# For GPU (recommended):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Verify Configuration

```bash
python config_zombie_sprint.py
```

This will display the complete configuration and setup instructions.

### 3. Setup the Game

1. Open [Zombie Sprint](https://www.onlinegames.io/zombie-sprint/)
2. Click "Choose Your Zombie"
3. Select **"Grave Sprinter"** (light zombie, good handling)
4. Click "Choose a Route"
5. Select **"Graveyard Loop"** (easiest track)
6. Set threat level to **"Easy"**
7. Position browser window in **TOP-LEFT corner** of screen
8. Ensure game area covers approximately 1000x800 pixels

### 4. Start Training

```bash
python train_zombie_sprint.py
```

The script will:
- Give you 10 seconds to position the game
- Start training automatically
- Save models every 100 episodes
- Display real-time visualization

---

## 📊 What Makes This Different

### Compared to Lane-Following Games

| Feature | Lane Following | Racing (Zombie Sprint) |
|---------|----------------|------------------------|
| **Objective** | Stay on road | Win race (beat opponents) |
| **Actions** | 7 (W, A, S, D, WA, WD, None) | 9 (+ Drift, Powerup) |
| **State** | Single frame | 4 stacked frames (motion) |
| **Rewards** | Lane centering | Position, speed, track boundaries |
| **Complexity** | Low | High |
| **Training Time** | 500-1000 episodes | 1500-2000 episodes |
| **Success Metric** | Stay on road | Finish 1st place |

---

## 🏗️ System Architecture

### Modified Components

#### 1. Actions (9 total)

```python
0: FORWARD               # W - Straight acceleration
1: FORWARD_LEFT          # W+A - Turn left while moving
2: FORWARD_RIGHT         # W+D - Turn right while moving
3: DRIFT_LEFT            # W+A+Space - Drift through left turns
4: DRIFT_RIGHT           # W+D+Space - Drift through right turns
5: BRAKE                 # S - Slow down/reverse
6: USE_POWERUP           # W+E - Use collected powerup
7: SLIGHT_LEFT           # W+A - Gentle left correction
8: SLIGHT_RIGHT          # W+D - Gentle right correction
```

**Key Additions**:
- **Drifting** (Space): Essential for maintaining speed through corners
- **Powerup** (E): Use items to gain advantage
- **Slight turns**: Finer steering control

#### 2. Frame Stacking

Unlike single-frame processing, racing requires **temporal information** (motion/speed):

```python
State = [Frame_t-3, Frame_t-2, Frame_t-1, Frame_t]
# 4 frames stacked together
# Allows agent to perceive movement and anticipate
```

#### 3. Reward System

**Multi-objective rewards**:

```python
Total Reward = Position Rewards 
             + Speed Rewards 
             + Track Staying Rewards 
             + Survival Rewards
```

**Position Rewards** (Most Important):
- 1st place: +10.0
- 2nd place: +5.0
- 3rd place: +3.0
- Improved position: +2.0
- Declined position: -2.0

**Speed Rewards**:
- Forward movement: +1.0
- High speed: +0.5
- Stopped: -1.0

**Track Rewards**:
- On track: +0.5
- Off track: -5.0
- Wrong direction: -3.0

**Special Bonuses**:
- Lap completed: +20.0
- Checkpoint passed: +5.0
- Race finished: +50.0

#### 4. Vision Processing

**Optimized for 3D racing**:

```python
# Original: Canny(threshold1=200, threshold2=300)
# Racing:   Canny(threshold1=100, threshold2=200)
# Reason: 3D environments need lower thresholds

# Original: No blur
# Racing:   GaussianBlur(7x7)
# Reason: Reduce noise from 3D rendering

# Original: Full image
# Racing:   ROI (Region of Interest) - bottom 80%
# Reason: Focus on track, ignore sky/scenery
```

#### 5. Neural Network

**Enhanced architecture**:

```python
Input: (4, 84, 84) - 4 stacked frames at 84x84 pixels

Conv1: 4 → 32 channels, 8x8 kernel, stride 4, BatchNorm, ReLU
Conv2: 32 → 64 channels, 4x4 kernel, stride 2, BatchNorm, ReLU
Conv3: 64 → 64 channels, 3x3 kernel, stride 1, BatchNorm, ReLU

Flatten

FC1: → 512, ReLU, Dropout(0.3)
FC2: 512 → 256, ReLU, Dropout(0.3)
FC3: 256 → 9 (Q-values for 9 actions)

Output: [Q(action_0), Q(action_1), ..., Q(action_8)]
```

**Key additions**:
- **Batch Normalization**: Stabilizes training
- **Dropout (0.3)**: Prevents overfitting to specific tracks

---

## 📈 Training Process

### Phase 1: Random Exploration (Episodes 1-100)

**What to expect**:
- Agent drives randomly
- Frequent crashes and off-road excursions
- Finishes in last place (7th-8th)
- Learns basic controls

**Metrics**:
- Average reward: 10-50
- Best position: 6th-8th
- Epsilon: 1.0 → 0.60

### Phase 2: Basic Racing (Episodes 101-500)

**What to expect**:
- Starts staying on track
- Completes laps occasionally
- Finishes in middle pack (4th-6th)
- Learns track layout

**Metrics**:
- Average reward: 50-150
- Best position: 4th-6th
- Epsilon: 0.60 → 0.37

### Phase 3: Competitive Racing (Episodes 501-1000)

**What to expect**:
- Consistent lap completion
- Competes for podium (1st-3rd)
- Uses drifting effectively
- Strategic overtaking

**Metrics**:
- Average reward: 150-300
- Best position: 2nd-4th
- Epsilon: 0.37 → 0.22

### Phase 4: Expert Racing (Episodes 1001-2000)

**What to expect**:
- Regular race wins
- Optimal racing lines
- Effective powerup usage
- Adapts to different tracks

**Metrics**:
- Average reward: 300-500+
- Best position: 1st-2nd
- Epsilon: 0.22 → 0.05

---

## ⚙️ Hyperparameter Tuning

### Learning Rate

```python
# Default: 0.0001
learning_rate = 0.0001

# Too high (>0.001): Unstable, oscillating rewards
# Too low (<0.00001): Extremely slow learning
# Optimal: 0.00005 - 0.0002
```

### Epsilon Decay

```python
# Default: 0.995 (gradual)
epsilon_decay = 0.995

# Faster (0.99): Exploits early, may miss strategies
# Slower (0.999): Explores longer, slower convergence
# For racing: 0.995-0.998 recommended (complex strategies need exploration)
```

### Batch Size

```python
# Default: 64
batch_size = 64

# Larger (128): More stable, requires more memory
# Smaller (32): Less stable, faster iteration
# For racing: 64-128 optimal
```

### Replay Memory

```python
# Default: 50,000 experiences
replay_memory_size = 50000

# Larger (100,000): More diverse, needs more RAM (4-8GB)
# Smaller (20,000): Less diverse, faster training
```

---

## 🎯 Training Tips

### 1. Character Selection

**Best for Training**:
- ✅ **Grave Sprinter** - Quick acceleration, sharp turns (RECOMMENDED)
- ✅ **Neon Ghoul** - Similar stats, good handling
- ✅ **Howler** - Fast acceleration

**Avoid Initially**:
- ❌ **Bone Crusher** - Heavy, slow turning (hard for AI)
- ❌ **Grave Queen** - Heavy, requires advanced control

### 2. Track Progression

**Curriculum Learning Approach**:

```
Episodes 1-500:    Graveyard Loop (Easy) + Easy Threat
Episodes 501-1000: Graveyard Loop (Easy) + Normal Threat
Episodes 1001-1500: Doomsday Highway (Normal) + Normal Threat
Episodes 1501-2000: Mix all tracks + Hard Threat
```

**Why this works**:
- Start simple, master basics
- Same track, harder opponents
- New tracks after fundamentals solid
- Final phase: generalization

### 3. Monitor Key Metrics

**During training, watch for**:

✅ **Good signs**:
- Reward steadily increasing
- Best position improving (8th → 6th → 4th → 1st)
- Epsilon decreasing smoothly
- Loss decreasing over time
- Episode length increasing (completes more laps)

❌ **Bad signs**:
- Reward oscillating wildly
- Always finishing last (8th)
- Loss increasing or NaN
- Agent goes backwards frequently
- Crashes immediately every episode

### 4. Reward Tuning

If agent has specific issues, adjust rewards:

**Problem: Agent drives off track constantly**
```python
# Increase penalties
REWARD_CONFIG['off_track'] = -10.0  # Was -5.0
REWARD_CONFIG['on_track'] = 1.0     # Was 0.5
```

**Problem: Agent drives too slowly**
```python
# Increase speed rewards
REWARD_CONFIG['forward_movement'] = 2.0  # Was 1.0
REWARD_CONFIG['high_speed'] = 1.0        # Was 0.5
```

**Problem: Agent ignores position, just drives**
```python
# Increase position importance
REWARD_CONFIG['position_1st'] = 20.0  # Was 10.0
REWARD_CONFIG['position_improvement'] = 5.0  # Was 2.0
```

**Problem: Agent never uses drift**
```python
# In code, add drift-specific rewards
if action in [3, 4]:  # Drift actions
    reward += 0.5  # Encourage trying drift
```

---

## 🔧 Troubleshooting

### Issue: Training is too slow

**Solution 1**: Reduce frame size
```python
# In config_zombie_sprint.py:
VISION_CONFIG['resize_dims'] = (64, 64)  # Was (84, 84)
```

**Solution 2**: Reduce capture area
```python
CAPTURE_BBOX = (0, 40, 800, 640)  # Smaller area
```

**Solution 3**: Use GPU
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Agent always finishes last

**Solution 1**: Train longer (need at least 500 episodes)

**Solution 2**: Increase position rewards
```python
REWARD_CONFIG['position_improvement'] = 5.0  # Higher reward for overtaking
```

**Solution 3**: Reduce epsilon decay (more exploration)
```python
epsilon_decay = 0.998  # Was 0.995
```

### Issue: Agent crashes into walls

**Solution**: Increase off-track penalty
```python
REWARD_CONFIG['off_track'] = -10.0
REWARD_CONFIG['wall_collision'] = -5.0
```

### Issue: Can't detect race position

**Current implementation uses placeholder**. For real position detection:

**Option 1**: Use OCR
```bash
pip install pytesseract
# Then read bottom-left corner text
```

**Option 2**: Template matching
```python
# Save images of "1st", "2nd", etc.
# Match against screen region
```

**Option 3**: Train without position (use only track staying)
```python
# Focus rewards on track staying and speed
# Position detection not required
```

---

## 📊 Expected Results

### After 500 Episodes

- **Completion Rate**: 60-80% (finishes races)
- **Average Position**: 5th-6th place
- **Can**: Drive full laps, basic cornering
- **Cannot**: Drift well, strategic overtaking

### After 1000 Episodes

- **Completion Rate**: 90-95%
- **Average Position**: 3rd-4th place
- **Can**: Drift through corners, compete for podium
- **Cannot**: Consistently win, advanced strategies

### After 2000 Episodes

- **Completion Rate**: 95-99%
- **Average Position**: 1st-2nd place
- **Can**: Win races regularly, optimal lines, powerup usage
- **Cannot**: Adapt to completely new tracks instantly

---

## 🎓 Advanced Topics

### 1. Transfer Learning

Train on one track, transfer to others:

```python
# Train on Graveyard Loop
train_agent(num_episodes=1000)

# Load and continue on new track
agent.load_model('models/zombie_sprint_episode_1000.pth')
agent.epsilon = 0.3  # Some exploration for new track
# Change to Doomsday Highway in game
train_agent(num_episodes=500)
```

### 2. Multi-Track Training

Train on multiple tracks simultaneously:

```python
# Manually switch tracks every 100 episodes
# Or modify code to detect track and log separately
```

### 3. Curriculum Learning

Implement automated difficulty progression:

```python
# Start Easy, auto-increase threat level based on performance
if average_position < 4.0:  # Doing well
    # Increase difficulty
    print("Increase threat level to Normal")
```

### 4. Imitation Learning

Pre-train from human demonstrations:

```python
# Record human gameplay (state, action pairs)
# Pre-train network to imitate
# Then fine-tune with RL
```

---

## 📁 File Structure

```
zombie-sprint-rl/
├── config_zombie_sprint.py      # All configurations
├── train_zombie_sprint.py       # Training script
├── rl_model_racing.py           # DQN with frame stacking
├── reward_racing.py             # Racing reward system
├── direction_keys.py            # Keyboard control (existing)
├── ZOMBIE_SPRINT_GUIDE.md       # This guide
│
└── models/                      # Saved models
    ├── zombie_sprint_episode_100.pth
    ├── zombie_sprint_episode_500.pth
    └── zombie_sprint_final.pth
```

---

## 🎯 Success Criteria

### Minimum Viable Agent (500 episodes)
- ✅ Completes 3 laps consistently
- ✅ Stays on track 80% of time
- ✅ Finishes in top 6

### Competitive Agent (1000 episodes)
- ✅ Finishes on podium (top 3) regularly
- ✅ Uses drift effectively
- ✅ Overtakes opponents

### Expert Agent (2000 episodes)
- ✅ Wins races (1st place) 40%+ of time
- ✅ Optimal racing lines
- ✅ Strategic powerup usage

---

## 🤔 FAQ

**Q: Why 9 actions instead of continuous control?**
A: Discrete actions are easier to learn. Racing games work well with discrete (proven by DQN on Atari racing games).

**Q: Why frame stacking?**
A: Single frames don't show motion/speed. Stacked frames let agent perceive "I'm moving forward fast" vs "I'm stationary".

**Q: Can this work on other racing games?**
A: Yes! Adjust these files:
- `config_zombie_sprint.py` - Change capture area, rewards
- `reward_racing.py` - Adapt reward calculations
- Same architecture works for most kart racers

**Q: How long does training take?**
A: 
- CPU: 6-10 hours for 2000 episodes
- GPU (GTX 1060): 2-3 hours
- GPU (RTX 3070): 1-1.5 hours

**Q: Can it play against human players?**
A: Not yet - current version plays against AI opponents. For multiplayer, need network access and more complex state.

---

## 📚 Additional Resources

**Reinforcement Learning**:
- [Deep Q-Learning Paper](https://arxiv.org/abs/1312.5602)
- [Rainbow DQN](https://arxiv.org/abs/1710.02298) - Advanced improvements

**Racing AI**:
- [World Models for Racing](https://worldmodels.github.io/)
- [Gran Turismo Sophy](https://www.nature.com/articles/s41586-021-04357-7) - Sony's racing AI

**Frame Stacking**:
- Standard technique in Atari DQN
- Solves partial observability problem

---

## ✅ Checklist Before Training

- [ ] Game open at https://www.onlinegames.io/zombie-sprint/
- [ ] "Grave Sprinter" selected
- [ ] "Graveyard Loop" track selected
- [ ] Threat level set to "Easy"
- [ ] Browser in TOP-LEFT corner (1000x800 capture)
- [ ] Python dependencies installed
- [ ] GPU available (optional but recommended)
- [ ] `models/` directory exists
- [ ] At least 2-3 hours available for training

---

## 🚀 Start Training Now!

```bash
# Quick start
python train_zombie_sprint.py

# With custom settings (edit config_zombie_sprint.py first)
python train_zombie_sprint.py
```

**Good luck and happy racing! 🏁🧟**

---

*Last updated: 2024*
