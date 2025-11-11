#dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from typing import Tuple, Optional
import os

from src.rl.network import QNetwork


# Experience tuple for replay buffer
Experience = namedtuple(
    'Experience',
    ['state', 'action', 'reward', 'next_state', 'done']
)

class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> list:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_size: int = 72,
        action_size: int = 4672,
        hidden_sizes: list = [512, 256, 128],
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: Optional[str] = None
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize networks
        self.q_network = QNetwork(
            input_size=state_size,
            hidden_sizes=hidden_sizes,
            output_size=action_size
        ).to(self.device)
        
        self.target_network = QNetwork(
            input_size=state_size,
            hidden_sizes=hidden_sizes,
            output_size=action_size
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # Training counters
        self.steps = 0
        self.episodes = 0
    
    def select_action(self, state: np.ndarray, legal_moves_count: int,
                     training: bool = True) -> int:
        # Exploration
        if training and random.random() < self.epsilon:
            return random.randint(0, legal_moves_count - 1)
        
        # Exploitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).flatten().unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            
            # Only consider legal moves
            q_values = q_values[0, :legal_moves_count]
            action = q_values.argmax().item()
        
        return action
    
    def train_step(self) -> Optional[float]:
        # Check if enough experiences
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Unpack experiences
        states = torch.FloatTensor(
            np.array([e.state.flatten() for e in experiences])
        ).to(self.device)
        
        actions = torch.LongTensor(
            [e.action for e in experiences]
        ).unsqueeze(1).to(self.device)
        
        rewards = torch.FloatTensor(
            [e.reward for e in experiences]
        ).unsqueeze(1).to(self.device)
        
        next_states = torch.FloatTensor(
            np.array([e.next_state.flatten() for e in experiences])
        ).to(self.device)
        
        dones = torch.FloatTensor(
            [float(e.done) for e in experiences]
        ).unsqueeze(1).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        
        print(f"Checkpoint loaded from {filepath}")
        print(f"Resumed at episode {self.episodes}, epsilon {self.epsilon:.4f}")