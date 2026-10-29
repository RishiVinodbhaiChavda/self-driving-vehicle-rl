# 🚀 Quick Start Guide

## Prerequisites Checklist

Before you start training, make sure you have:

- [ ] Python 3.8+ installed
- [ ] All dependencies installed
- [ ] A driving game ready (recommended: Fumes)
- [ ] Game positioned in TOP-LEFT corner of screen

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you have a NVIDIA GPU, install PyTorch with CUDA for faster training:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For CPU only:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 2: Verify Installation

Run this to check if everything is installed:

```bash
python -c "import torch; import cv2; import numpy; print('All dependencies installed!')"
```

## Training

### Method 1: Recommended Game (Fumes)

1. **Open the game**: Go to [playfumes.com](https://playfumes.com/) in your browser
2. **Position the browser**: Move it to the TOP-LEFT corner of your screen
3. **Resize the window**: Make sure the game area covers approximately 800x800 pixels
4. **Press Space** to start the game (the car should be on the road)
5. **Start training**:

```bash
python train_rl.py
```

6. You'll see:
   - A 5-second countdown
   - Then the training window will open showing what the AI sees
   - Console output showing episode progress

### Method 2: Other Browser Racing Games

You can try:
- **Racing Limits**: [kbhgames.com/game/racing-limits](https://kbhgames.com/game/racing-limits)
- **Racing Unlimited**: [crazygames.com/game/racing-unlimited](https://www.crazygames.com/game/racing-unlimited)

Same steps as above!

## Training Tips

### During Training

- **Don't move the game window!** The AI captures a fixed screen region
- **Let it run for at least 100-200 episodes** to see meaningful learning
- **Early episodes will look random** - that's normal (exploration phase)
- **Press 'q' in the OpenCV window** to stop training anytime
- **Models are auto-saved** every 50 episodes in the `models/` folder

### What You'll See

```
[Episode 1/1000] Starting...
[Episode 1] Finished!
  Steps: 45 | Total Reward: 12.30
  Avg Loss: 0.0234 | Epsilon: 0.995
  Avg Reward (last 10): 12.30
  Frames Survived: 45
```

- **Steps**: How many frames before crashing/ending
- **Total Reward**: Higher is better
- **Epsilon**: Exploration rate (starts at 1.0, decreases to 0.01)
- **Avg Loss**: Should decrease over time

### Expected Progress

| Episode Range | What to Expect |
|---------------|----------------|
| 1-50 | Random driving, lots of crashes, learning basic controls |
| 50-200 | Starts staying on road longer, fewer instant crashes |
| 200-500 | Can drive straight, basic steering working |
| 500-1000 | Smooth driving, good lane keeping, strategic decisions |

## Testing Your Trained Model

After training (or during - models are saved periodically):

```bash
python test_rl.py models/dqn_model_final.pth
```

Or test a specific checkpoint:

```bash
python test_rl.py models/dqn_model_episode_500.pth
```

Same positioning rules apply - game in TOP-LEFT corner!

## Troubleshooting

### Issue: "No module named 'torch'"

**Solution**: Install PyTorch
```bash
pip install torch torchvision
```

### Issue: Training stops immediately / No episodes completed

**Solution**: 
- Make sure the game is actually running and visible
- Check that the game window is in the TOP-LEFT corner
- Verify the capture region by running the old `pds.py` first to see if it captures correctly

### Issue: Agent always crashes instantly

**Solution**:
- Train for more episodes (need at least 100-200 to learn)
- Adjust reward thresholds in `reward_system.py`
- Try a simpler game like Fumes

### Issue: "CUDA out of memory"

**Solution**:
- Reduce batch size in `train_rl.py`: change `batch_size=32` to `batch_size=16`
- Or use CPU by installing CPU version of PyTorch

### Issue: Training is very slow

**Solution**:
- Install CUDA version of PyTorch for GPU acceleration
- Reduce max_steps_per_episode: change `max_steps_per_episode=1000` to `500`
- Close other programs to free up resources

### Issue: Can't see the training window

**Solution**:
- Check if it's behind other windows
- Make sure `cv2` window isn't minimized
- Try pressing Alt+Tab to find it

## Quick Test Before Training

Want to test if everything works? Run the old script first:

```bash
python pds.py
```

This will show you what the AI "sees". If you see a window with detected edges and lines from your game, you're good to go!

Press 'q' to quit, then run the training script.

## Controls During Training/Testing

- **'q' key** - Stop training/testing
- **Ctrl+C** - Emergency stop (in terminal)
- **DO NOT** click on the game window during training - let the AI control it

## File Structure

After training, you'll have:

```
Self_drive_vehicle-master/
├── models/                          # Saved models
│   ├── dqn_model_episode_50.pth    # Checkpoint at episode 50
│   ├── dqn_model_episode_100.pth   # Checkpoint at episode 100
│   ├── dqn_model_episode_150.pth   # etc...
│   └── dqn_model_final.pth         # Final model
├── train_rl.py                      # Training script
├── test_rl.py                       # Testing script
├── rl_model.py                      # Neural network
├── reward_system.py                 # Reward calculations
└── ...
```

## Advanced: Customizing Training

Edit `train_rl.py` at the bottom:

```python
train_agent(
    num_episodes=1000,           # Change to train more/less
    max_steps_per_episode=1000,  # Max frames per episode
    batch_size=32,               # Training batch size
    target_update_freq=10,       # Update target network frequency
    save_freq=50                 # Save model every X episodes
)
```

## Performance Benchmarks

On a typical system:

- **With GPU (CUDA)**: ~10-20 episodes/minute
- **With CPU only**: ~3-5 episodes/minute

Training 1000 episodes:
- **With GPU**: ~1-2 hours
- **With CPU**: ~3-5 hours

## Getting Help

If you're stuck:

1. Read the error message carefully
2. Check the Troubleshooting section above
3. Make sure game is positioned correctly
4. Try the simpler test script (`pds.py`) first
5. Verify all dependencies are installed

## Ready to Train?

1. ✅ Install dependencies
2. ✅ Open game (Fumes recommended)
3. ✅ Position in TOP-LEFT corner
4. ✅ Run: `python train_rl.py`
5. ✅ Wait and watch it learn!

Good luck! 🚗💨
