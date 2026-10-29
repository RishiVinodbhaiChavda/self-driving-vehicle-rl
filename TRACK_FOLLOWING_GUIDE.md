# 🛣️ Track Color Following Guide

## Your Specific Game - Perfect Use Case!

Based on your screenshot, this system is **IDEAL** for your game:
- ✅ **Gray road** with clear boundaries
- ✅ **Green grass** as off-track area
- ✅ **White lane markings** visible
- ✅ **Realistic 3D driving simulator**

---

## 🎯 The Binary Approach

### Simple Concept

```
Color Detection → Binary Mask:
- ROAD (gray) = WHITE pixels
- GRASS (green) = BLACK pixels

AI learns: "Follow the white, avoid the black"
```

### Why This Is Better for Your Game

| Old Method (Edge Detection) | New Method (Color Binary) |
|----------------------------|---------------------------|
| Detects ALL edges | Detects ONLY road |
| Confused by barriers | Ignores barriers |
| Complex processing | Simple & fast |
| Hard to tune | Easy to understand |

---

## 🚀 Quick Start (3 Steps)

### Step 1: Test the Detection

```bash
python track_color_following.py
```

**What you'll see**:
- **Left window**: Your game
- **Right window**: Binary mask (WHITE=road, BLACK=grass)
- **Green overlay**: Detected drivable area
- **Magenta line**: Your car position
- **Yellow line**: Track center
- **Green arrow**: Where to steer

**Controls**:
- `q` - Quit
- `c` - Calibrate colors for your game
- `b` - Toggle binary view

### Step 2: Verify It Works

Watch the visualization:
- ✅ Road should be WHITE
- ✅ Grass should be BLACK
- ✅ "ON TRACK" when car on road
- ✅ "OFF TRACK!" when car on grass

If colors are wrong, press `c` to calibrate!

### Step 3: Train

```bash
python train_track_following.py
```

The AI will:
1. Learn to stay on white (road)
2. Avoid black (grass)
3. Auto-press 'R' if goes off-track
4. Get penalty for going off-road

---

## 🎮 Game Setup

### Important Settings

**Capture Area**: `(0, 40, 1024, 768)`
- Top-left: (0, 40)
- Bottom-right: (1024, 768)
- **Make sure game window covers this area!**

**Position your game**:
1. Open your driving game
2. Move window to TOP-LEFT corner of screen
3. Game should start at coordinates (0, 40)
4. Visible area should be ~1024x768 pixels

---

## 🔧 How It Works

### Color Detection

```python
# ROAD (Gray/Brown from your game)
H: 0-180   (any hue - gray has no color)
S: 0-60    (low saturation = grayish)
V: 40-130  (medium brightness)

# GRASS (Green)
H: 35-85   (green hues)
S: 30-255  (some saturation)
V: 40-255  (visible green)

# WHITE LANE MARKINGS
H: 0-180   (any hue)
S: 0-30    (very low saturation)
V: 180-255 (very bright)
```

### Binary Mask Creation

```
1. Convert image to HSV
2. Apply color range thresholds
3. Create masks: road_mask, grass_mask, white_mask
4. Combine: drivable = road OR white
5. Clean noise with morphology
6. Output: WHITE = drivable, BLACK = off-track
```

### Off-Track Detection

```python
# Sample region where car is (bottom-center)
car_region = image[70%-90% height, 40%-60% width]

# Calculate road percentage
road_pixels = count_white_pixels(car_region)
road_percentage = road_pixels / total_pixels

# Threshold
if road_percentage < 30%:
    OFF_TRACK = True
    press_r_key()  # Reset game
    penalty = -10.0
```

### Steering Calculation

```python
# Look ahead (50% height)
scan_line = binary_mask[50% height, :]

# Find track edges
left_edge = first_white_pixel
right_edge = last_white_pixel
track_center = (left_edge + right_edge) / 2

# Calculate offset from image center
offset = (track_center - image_center) / (width / 2)

# offset: -1.0 (far left) to +1.0 (far right)
```

---

## 🎓 Reward System

### On-Track Rewards

```python
# Base reward for staying on road
+1.0  - Being on track

# Centering bonus
+2.0 × (1.0 - |offset|)  - The more centered, the higher

# Forward movement
+0.5  - Moving forward (assumed when on track)

# Survival
+0.1  - Each frame

# Milestone
+5.0  - Every 100 frames on track
```

### Off-Track Penalties

```python
# Off-track penalty
-10.0  - Being on grass

# Reset penalty
-20.0  - Caused game to reset (5+ consecutive off-track frames)

# No positive rewards when off-track
```

### Total Reward Range

- **Best case**: ~3.6 per frame (perfectly centered, on track)
- **Worst case**: -20.0 (caused reset)
- **Average good driving**: ~1.5 per frame

---

## 📊 Training Expectations

### Episode 0-100 (Learning Basics)
- Random driving
- Frequent off-track
- Many resets (~20-30 per episode)
- Reward: -50 to +50
- **Goal**: Learn that white = good, black = bad

### Episode 100-300 (Staying On Track)
- Starts staying on road
- Fewer resets (~5-10 per episode)
- Reward: +50 to +200
- **Goal**: Avoid grass consistently

### Episode 300-600 (Centering)
- Good track staying
- Learning to stay centered
- Minimal resets (~1-3 per episode)
- Reward: +200 to +500
- **Goal**: Stay in middle of lane

### Episode 600-1000 (Mastery)
- Excellent driving
- Rare resets (~0-1 per episode)
- Reward: +500 to +1000
- **Goal**: Smooth, centered driving

---

## 🔧 Calibration

### Auto-Calibration

If colors don't match:

```bash
python track_color_following.py
# Press 'c' when game is visible
# System will analyze colors and suggest ranges
```

### Manual Calibration

Edit `track_color_following.py`:

```python
# Adjust these ranges in process_track_binary():

# ROAD
lower_road = np.array([0, 0, 40])    # [H, S, V]
upper_road = np.array([180, 60, 130])

# GRASS
lower_grass = np.array([35, 30, 40])
upper_grass = np.array([85, 255, 255])

# WHITE MARKINGS
lower_white = np.array([0, 0, 180])
upper_white = np.array([180, 30, 255])
```

**Tips**:
- Increase ranges if detection is too strict
- Decrease ranges if detecting too much
- Use calibration tool to find optimal values

---

## 🐛 Troubleshooting

### Problem: Road not detected (all black)

**Cause**: HSV ranges too strict  
**Fix**: Widen the ranges
```python
lower_road = np.array([0, 0, 20])    # Lower min value
upper_road = np.array([180, 80, 150]) # Higher max
```

### Problem: Grass detected as road (grass is white)

**Cause**: Ranges overlap  
**Fix**: Narrow road ranges, exclude green
```python
# Make sure grass detection is good
lower_grass = np.array([30, 40, 40])  # Stricter green
upper_grass = np.array([90, 255, 255])
```

### Problem: Always off-track even when on road

**Cause**: Threshold too high  
**Fix**: Lower the threshold
```python
# In train_track_following.py:
env = TrackFollowingEnvironment(
    off_track_threshold=0.2,  # Was 0.3 (30%)
    reset_on_off_track=True
)
```

### Problem: 'R' key not resetting game

**Cause**: Game uses different reset key  
**Fix**: Change the key code
```python
# In track_color_following.py:
R_KEY = 0x13  # Change to your game's reset key
# Find key codes: https://docs.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes
```

### Problem: Too many resets during training

**Cause**: Agent learning, expected in early episodes  
**Fix**: 
- Lower off-track threshold temporarily
- Increase consecutive frame requirement
```python
# In train_track_following.py, step():
if self.off_track_count >= 10:  # Was 5, now need 10 consecutive
    # Reset
```

---

## 📈 Advanced Tips

### 1. Gradual Difficulty

Start easy, increase difficulty:

```python
# Episodes 0-300: Loose threshold
off_track_threshold = 0.2

# Episodes 300-600: Medium threshold  
off_track_threshold = 0.3

# Episodes 600+: Strict threshold
off_track_threshold = 0.4
```

### 2. Look-Ahead Distance

Adjust where agent looks:

```python
# In calculate_track_center():
lookahead_y = int(height * 0.4)  # Look further ahead (was 0.5)
# Further = more planning, closer = more reactive
```

### 3. Speed Control

Add speed estimation:

```python
# Count white pixels between frames
current_white = count_white(frame_t)
prev_white = count_white(frame_t-1)
speed_estimate = abs(current_white - prev_white)

# Reward high speed when on track
if on_track and speed_estimate > threshold:
    reward += 0.5
```

---

## 📁 Files

```
track-following/
├── track_color_following.py    # Detection & visualization
├── train_track_following.py    # RL training script
└── TRACK_FOLLOWING_GUIDE.md    # This guide
```

---

## ✅ Checklist Before Training

- [ ] Game open and running
- [ ] Game window in TOP-LEFT corner
- [ ] Tested with `python track_color_following.py`
- [ ] Binary mask looks correct (white=road)
- [ ] "ON TRACK" shows when car is on road
- [ ] "OFF TRACK" shows when car is on grass
- [ ] Calibrated colors if needed (press 'c')
- [ ] 'R' key successfully resets game
- [ ] Ready to train!

---

## 🎯 Summary

**Your Game**: Perfect for this approach  
**Method**: Binary color detection (WHITE=road, BLACK=grass)  
**Advantage**: Simple, fast, accurate  
**Auto-Reset**: Presses 'R' when off-track  
**Penalty**: -10 for off-track, -20 for reset  

**Start**: `python track_color_following.py` (test)  
**Train**: `python train_track_following.py` (learn)  
**Result**: AI that stays on road and avoids grass!

---

**Perfect solution for your specific game!** 🎮✨

*Binary approach = Simple + Effective!* ✅
