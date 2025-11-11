#eval_utils.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime


def evaluate_agent(env, agent, episodes: int = 10, verbose: bool = True) -> Dict[str, float]:
    rewards = []
    move_counts = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False
        moves = 0
        
        while not done:
            # Select action without exploration (greedy policy)
            legal_moves_count = env.get_legal_moves_count()
            action = agent.select_action(state, legal_moves_count, training=False)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            moves += 1
            state = next_state
        
        rewards.append(episode_reward)
        move_counts.append(moves)
        
        if verbose:
            print(f"Eval Episode {episode + 1}/{episodes}: "
                  f"Reward = {episode_reward:.4f}, Moves = {moves}")
    
    metrics = {
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'avg_moves': np.mean(move_counts),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards)
    }
    
    if verbose:
        print("\n--- Evaluation Summary ---")
        print(f"Average Reward: {metrics['avg_reward']:.4f} ± {metrics['std_reward']:.4f}")
        print(f"Reward Range: [{metrics['min_reward']:.4f}, {metrics['max_reward']:.4f}]")
        print(f"Average Moves: {metrics['avg_moves']:.2f}")
    
    return metrics

def save_example_game(env, agent, filepath: str, seed: Optional[int] = None):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write("=== Chess960 RL Agent Example Game ===\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if seed is not None:
            f.write(f"Seed: {seed}\n")
        f.write("\n")
        
        state = env.reset(seed=seed)
        f.write("Initial Position:\n")
        f.write(env.render())
        f.write("\n\n")
        
        done = False
        move_num = 1
        
        while not done:
            # Select action
            legal_moves_count = env.get_legal_moves_count()
            action = agent.select_action(state, legal_moves_count, training=False)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Record move
            turn = "White" if move_num % 2 == 1 else "Black"
            f.write(f"Move {move_num} ({turn}): {info['last_move']}\n")
            
            if done:
                f.write(f"\nFinal Reward: {reward:.4f}\n")
                f.write("\nFinal Position:\n")
                f.write(env.render())
            
            state = next_state
            move_num += 1
    
    print(f"Example game saved to {filepath}")

def plot_training_metrics(
    rewards: List[float],
    losses: List[float],
    epsilon_values: List[float],
    save_path: str = "training_metrics.png",
    window_size: int = 50
):

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Helper function for moving average
    def moving_average(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Plot 1: Episode Rewards
    ax1 = axes[0]
    ax1.plot(rewards, alpha=0.3, color='blue', label='Raw Rewards')
    if len(rewards) >= window_size:
        smoothed_rewards = moving_average(rewards, window_size)
        ax1.plot(range(window_size-1, len(rewards)), smoothed_rewards, 
                color='darkblue', linewidth=2, label=f'{window_size}-Episode MA')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training Loss
    ax2 = axes[1]
    if losses:
        ax2.plot(losses, alpha=0.3, color='red', label='Raw Loss')
        if len(losses) >= window_size:
            smoothed_losses = moving_average(losses, window_size)
            ax2.plot(range(window_size-1, len(losses)), smoothed_losses,
                    color='darkred', linewidth=2, label=f'{window_size}-Step MA')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')  # Log scale often better for loss
    
    # Plot 3: Exploration Rate (Epsilon)
    ax3 = axes[2]
    ax3.plot(epsilon_values, color='green', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('Exploration Rate Decay')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training metrics plot saved to {save_path}")
    plt.close()


def plot_reward_distribution(rewards: List[float], save_path: str = "reward_distribution.png"):
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(rewards), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(rewards):.4f}')
    plt.axvline(np.median(rewards), color='orange', linestyle='--',
                linewidth=2, label=f'Median: {np.median(rewards):.4f}')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution of Episode Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Reward distribution plot saved to {save_path}")
    plt.close()


def log_training_progress(
    episode: int,
    reward: float,
    epsilon: float,
    loss: Optional[float],
    avg_reward: float,
    log_file: str = "training.log"
):
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
    
    with open(log_file, 'a') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        loss_str = f"{loss:.6f}" if loss is not None else "N/A"
        f.write(f"[{timestamp}] Episode {episode:5d} | "
                f"Reward: {reward:7.4f} | "
                f"Avg: {avg_reward:7.4f} | "
                f"Epsilon: {epsilon:.4f} | "
                f"Loss: {loss_str}\n")


def create_training_summary(
    total_episodes: int,
    final_metrics: Dict[str, float],
    training_time: float,
    save_path: str = "training_summary.txt"
):
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Chess960 RL Training Summary\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total Episodes: {total_episodes}\n")
        f.write(f"Training Time: {training_time/3600:.2f} hours\n")
        f.write(f"Episodes per Hour: {total_episodes/(training_time/3600):.1f}\n\n")
        
        f.write("Final Evaluation Metrics:\n")
        f.write("-" * 40 + "\n")
        for key, value in final_metrics.items():
            f.write(f"{key:20s}: {value:.4f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"Training summary saved to {save_path}")