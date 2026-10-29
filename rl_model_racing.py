"""
Deep Q-Network for Racing Games
Adapted for frame stacking and racing-specific actions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from config_zombie_sprint import NETWORK_CONFIG, STATE_CONFIG, TRAINING_CONFIG


class DQNRacing(nn.Module):
    """
    DQN for racing with frame stacking
    Input: Stacked frames (num_frames, height, width)
    Output: Q-values for racing actions
    """
    def __init__(self, num_actions):
        super(DQNRacing, self).__init__()
        
        # Input: (batch, num_frames, height, width)
        input_channels = STATE_CONFIG['num_frames']
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Calculate conv output size
        # For (4, 84, 84) input
        conv_output_size = self._get_conv_output_size(
            STATE_CONFIG['frame_size'][0], 
            STATE_CONFIG['frame_size'][1]
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.dropout1 = nn.Dropout(NETWORK_CONFIG['dropout'])
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(NETWORK_CONFIG['dropout'])
        
        self.fc3 = nn.Linear(256, num_actions)
        
    def _get_conv_output_size(self, h, w):
        """Calculate output size after convolutions"""
        # Conv1: kernel=8, stride=4
        h = (h - 8) // 4 + 1
        w = (w - 8) // 4 + 1
        # Conv2: kernel=4, stride=2
        h = (h - 4) // 2 + 1
        w = (w - 4) // 2 + 1
        # Conv3: kernel=3, stride=1
        h = (h - 3) // 1 + 1
        w = (w - 3) // 1 + 1
        return 64 * h * w
    
    def forward(self, x):
        """Forward pass"""
        # x shape: (batch, num_frames, height, width)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


class ReplayMemory:
    """Experience replay memory"""
    def __init__(self, capacity=50000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Save a transition"""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Randomly sample a batch"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class DQNRacingAgent:
    """
    DQN Agent for racing games
    """
    def __init__(self, num_actions=9, learning_rate=0.0001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        """
        Initialize racing agent
        
        Args:
            num_actions: Number of racing actions
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration
            epsilon_end: Minimum exploration
            epsilon_decay: Decay rate
        """
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Networks
        self.policy_net = DQNRacing(num_actions).to(self.device)
        self.target_net = DQNRacing(num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay memory
        self.memory = ReplayMemory(capacity=TRAINING_CONFIG['replay_memory_size'])
        
        # Training step counter
        self.train_step_counter = 0
        
    def select_action(self, state):
        """
        Select action using epsilon-greedy
        
        Args:
            state: Stacked frames (num_frames, height, width)
            
        Returns:
            action_idx: Selected action
        """
        # Exploration
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        
        # Exploitation
        with torch.no_grad():
            # Add batch dimension and convert to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay memory"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self, batch_size=64):
        """
        Perform one training step
        
        Args:
            batch_size: Batch size
            
        Returns:
            loss: Training loss or None
        """
        if len(self.memory) < batch_size:
            return None
        
        # Sample batch
        transitions = self.memory.sample(batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch_state)).to(self.device)
        action_batch = torch.LongTensor(batch_action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch_reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch_next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch_done).unsqueeze(1).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        self.train_step_counter += 1
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath):
        """Save model to disk"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step_counter': self.train_step_counter
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from disk"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_step_counter = checkpoint.get('train_step_counter', 0)
        print(f"Model loaded from {filepath}")
