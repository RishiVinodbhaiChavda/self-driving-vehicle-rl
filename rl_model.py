"""
Reinforcement Learning Model for Self-Driving Vehicle
Uses Deep Q-Network (DQN) to learn driving behavior
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random


class DQN(nn.Module):
    """
    Deep Q-Network for learning driving actions
    Input: Processed screen image (grayscale)
    Output: Q-values for each action [W, A, S, D, WA, WD, No Action]
    """
    def __init__(self, input_channels=1, num_actions=7):
        super(DQN, self).__init__()
        
        # Convolutional layers to process screen input
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size after convolutions
        # For input size (760, 760) -> after convolutions
        conv_output_size = self._get_conv_output_size(760, 760)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
    def _get_conv_output_size(self, h, w):
        """Calculate the output size after convolutions"""
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
        """Forward pass through the network"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ReplayMemory:
    """
    Experience Replay Memory to store and sample transitions
    Helps break correlation between consecutive samples
    """
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Save a transition"""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Randomly sample a batch of transitions"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """
    DQN Agent that interacts with the environment and learns
    """
    def __init__(self, num_actions=7, learning_rate=0.0001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        """
        Initialize the DQN Agent
        
        Args:
            num_actions: Number of possible actions (7: W, A, S, D, WA, WD, NoAction)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Rate at which exploration decreases
        """
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create policy network and target network
        self.policy_net = DQN(num_actions=num_actions).to(self.device)
        self.target_net = DQN(num_actions=num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is not trained directly
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay memory
        self.memory = ReplayMemory(capacity=10000)
        
        # Action mapping
        self.actions = {
            0: 'W',      # Forward
            1: 'A',      # Left
            2: 'S',      # Backward
            3: 'D',      # Right
            4: 'WA',     # Forward + Left
            5: 'WD',     # Forward + Right
            6: 'NONE'    # No action
        }
        
    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy
        
        Args:
            state: Current state (processed screen image)
            
        Returns:
            action_idx: Index of the selected action
        """
        # Exploration: random action
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        
        # Exploitation: best action according to policy network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in replay memory"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self, batch_size=32):
        """
        Perform one training step
        
        Args:
            batch_size: Number of samples to train on
            
        Returns:
            loss: Training loss (or None if not enough samples)
        """
        if len(self.memory) < batch_size:
            return None
        
        # Sample a batch from memory
        transitions = self.memory.sample(batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch_state)).to(self.device)
        action_batch = torch.LongTensor(batch_action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch_reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch_next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch_done).unsqueeze(1).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath):
        """Save the model to disk"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load the model from disk"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filepath}")
