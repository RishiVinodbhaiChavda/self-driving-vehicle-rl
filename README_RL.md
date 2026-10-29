# Self-Driving Vehicle with Reinforcement Learning

A complete implementation of a self-driving vehicle AI using Deep Q-Network (DQN) reinforcement learning to play driving games autonomously.

## 🎯 What This Does

This project uses reinforcement learning to teach an AI agent to drive in video games by:
1. **Capturing** the game screen in real-time
2. **Processing** the image to detect road lanes using computer vision
3. **Learning** optimal driving actions through trial and error
4. **Controlling** the vehicle by simulating keyboard inputs (W, A, S, D)

## 🧠 How It Works

### Components

1. **rl_model.py** - Deep Q-Network (DQN) implementation
   - Neural network that learns to predict the best action for each state
   - Experience replay memory to store and learn from past experiences
   - Target network for stable training

2. **reward_system.py** - Reward calculation
   - Rewards for staying on the road
   - Rewards for staying centered in the lane
   - Rewards for moving forward
   - Penalties for crashing or going off-road

3. **train_rl.py** - Training script
   - Combines all components
   - Runs training episodes
   - Saves models periodically

4. **test_rl.py** - Testing script
   - Loads a trained model
   - Watches the AI drive

### Actions Available

The agent can choose from 7 actions:
- **W** - Forward
- **A** - Left
- **S** - Backward (brake)
- **D** - Right
- **W+A** - Forward + Left (turn left while moving)
- **W+D** - Forward + Right (turn right while moving)
- **No Action** - Coast

### Reinforcement Learning Algorithm

**Deep Q-Network (DQN)**:
- Takes the processed screen as input (grayscale image with detected edges)
- Outputs Q-values for each action (expected future reward)
- Selects the action with highest Q-value
- Learns by minimizing the difference between predicted and actual rewards

**Key Concepts**:
- **Exploration vs Exploitation**: Agent starts by exploring random actions (ε=1.0) and gradually exploits learned knowledge (ε→0.01)
- **Experience Replay**: Stores past experiences and trains on random batches to break correlation
- **Target Network**: Stabilizes training by keeping a separate network for computing target values

## 📦 Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

**Note**: PyTorch installation might require specific commands based on your system. Visit [pytorch.org](https://pytorch.org) for details.

For CPU-only:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

For CUDA (GPU):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 Usage

### Training the Agent

1. **Start your driving game** (GTA, Need for Speed, or any racing game)
2. **Position the game window** in the TOP-LEFT corner of your screen
3. **Run the training script**:

```bash
python train_rl.py
```

The agent will start learning! You'll see:
- A window showing what the AI sees (processed edges and lanes)
- Real-time statistics (episode, reward, epsilon, etc.)
- Model saves every 50 episodes in the `models/` folder

**Training Parameters** (can be adjusted in `train_rl.py`):
- `num_episodes=1000` - Total training episodes
- `max_steps_per_episode=1000` - Maximum steps per episode
- `batch_size=32` - Training batch size
- `target_update_freq=10` - How often to update target network
- `save_freq=50` - How often to save the model

**Stop training**: Press 'q' in the OpenCV window

### Testing the Agent

After training, test your trained model:

```bash
python test_rl.py models/dqn_model_final.pth
```

Or test a specific checkpoint:

```bash
python test_rl.py models/dqn_model_episode_500.pth
```

The AI will drive autonomously! Watch it in action.

**Stop testing**: Press 'q'

## 📊 Understanding the Output

### During Training

```
[Episode 1/1000] Starting...
[Episode 1] Finished!
  Steps: 245 | Total Reward: 78.50
  Avg Loss: 0.0234 | Epsilon: 0.995
  Avg Reward (last 10): 78.50
  Frames Survived: 245
```

- **Steps**: How many frames the agent survived before crashing
- **Total Reward**: Cumulative reward for the episode (higher is better)
- **Avg Loss**: Training loss (should decrease over time)
- **Epsilon**: Exploration rate (decreases from 1.0 to 0.01)
- **Frames Survived**: Same as steps, but more intuitive name

### Reward System

The agent gets:
- **+2.0** for staying near center of lane
- **+1.0** for detecting good road lines
- **+0.1** for each frame survived
- **+5.0** bonus every 100 frames
- **-10.0** for crashing or going off-road

## 🎮 Tips for Best Results

1. **Game Selection**: 
   - Use games with clear lane markings
   - Simpler tracks work better initially
   - GTA V and racing games work well

2. **Screen Position**:
   - Game MUST be in top-left corner
   - Make sure the road is visible
   - Adjust brightness/contrast for clear lanes

3. **Training Duration**:
   - Let it train for at least 500-1000 episodes
   - Early episodes will be random (exploration)
   - Performance improves gradually

4. **Hyperparameter Tuning**:
   - Increase `learning_rate` if learning is too slow
   - Decrease `epsilon_decay` for more exploration
   - Adjust reward values in `reward_system.py`

## 🔧 Customization

### Modify Rewards

Edit `reward_system.py` to change how the agent is rewarded:

```python
def calculate_reward(self, processed_img, original_img):
    # Add your custom rewards here
    if custom_condition:
        reward += custom_value
```

### Change Actions

Edit `rl_model.py` and `train_rl.py` to add/remove actions:

```python
self.actions = {
    0: 'W',
    1: 'A',
    # Add more actions...
}
```

### Adjust Network Architecture

Modify the DQN architecture in `rl_model.py`:

```python
def __init__(self, input_channels=1, num_actions=7):
    # Modify layers here
    self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
    # Add more layers...
```

## 📈 Expected Training Progress

- **Episodes 0-100**: Random driving, frequent crashes, learning basic survival
- **Episodes 100-300**: Starts staying on road, basic steering
- **Episodes 300-600**: Smoother driving, lane centering improves
- **Episodes 600-1000**: Confident driving, good lane keeping

## ⚠️ Known Limitations

1. **Screen capture only**: Only works with games running on your screen
2. **Fixed screen region**: Game must be positioned correctly
3. **Simple vision**: Only uses edge detection, not full visual understanding
4. **No speedometer**: Speed estimation is approximate
5. **Windows only**: Keyboard control uses Windows API

## 🔮 Future Improvements

- [ ] Add multi-lane detection
- [ ] Implement traffic awareness
- [ ] Add support for complex intersections
- [ ] Train on multiple games
- [ ] Add visualization of Q-values
- [ ] Implement Prioritized Experience Replay
- [ ] Try other RL algorithms (A3C, PPO, SAC)

## 🐛 Troubleshooting

**Problem**: Agent keeps crashing immediately
- **Solution**: Adjust reward thresholds in `reward_system.py`, train longer

**Problem**: Training is very slow
- **Solution**: Install CUDA version of PyTorch for GPU acceleration

**Problem**: Can't find lanes
- **Solution**: Adjust Canny thresholds and Hough parameters in `process_img()`

**Problem**: Actions not working in game
- **Solution**: Make sure game window is focused and accepts keyboard input

## 📚 Learn More

- [Deep Q-Network Paper](https://www.nature.com/articles/nature14236)
- [Reinforcement Learning Intro](https://spinningup.openai.com/)
- [PyTorch Tutorial](https://pytorch.org/tutorials/)

## 🤝 Contributing

Feel free to improve this project! Some ideas:
- Better reward functions
- Different RL algorithms
- Support for more games
- Better vision processing

---

**Enjoy teaching your AI to drive! 🚗💨**
