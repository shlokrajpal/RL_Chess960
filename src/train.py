#train.py

import os
import sys
import time
import numpy as np
from collections import deque
import chess

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.chess960_env import Chess960Env
from src.rl.dqn_agent import DQNAgent
from utils.eval_utils import (
    evaluate_agent,
    plot_training_metrics,
    plot_reward_distribution,
    log_training_progress,
    save_example_game,
    create_training_summary
)


def store_game_outcome(replay_buffer, game_history, final_reward_white_perspective):
    for state, action, next_state, is_white_turn in game_history:
        # Convert reward to the perspective of the player who made this move
        if is_white_turn:
            player_reward = final_reward_white_perspective
        else:
            # Flip for black: if white's reward is +0.8, black's is -0.8
            player_reward = -final_reward_white_perspective
        
        # Store as terminal experience (all moves led to this game outcome)
        replay_buffer.push(state, action, player_reward, next_state, done=True)


def train(
    # Training hyperparameters
    num_episodes: int = 1000,
    max_moves: int = 30,  # Can be any number now - color alternation handles balance
    eval_frequency: int = 50,
    eval_episodes: int = 10,
    
    # Agent hyperparameters
    learning_rate: float = 0.001,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    
    # Network hyperparameters
    hidden_sizes: list = [512, 256, 128],
    
    # Replay buffer
    buffer_capacity: int = 100000,
    batch_size: int = 64,
    min_replay_size: int = 1000,
    
    # Training parameters
    target_update_freq: int = 1000,
    train_frequency: int = 4,
    
    # Stockfish settings
    stockfish_path: str = r"D:\stockfish_17.1\stockfish\stockfish-windows-x86-64-avx2.exe",
    stockfish_depth: int = 15,
    
    # Saving and logging
    save_dir: str = "checkpoints",
    log_dir: str = "logs",
    save_frequency: int = 100,
    
    # Reward perspective (should always be "white" for self-play)
    reward_perspective: str = "white"
):
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Print training configuration
    print("=" * 70)
    print("Chess960 DQN Training - COLOR BALANCE FIXED")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Max moves per episode: {max_moves}")
    print(f"Learning rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"Epsilon: {epsilon_start} -> {epsilon_end} (decay: {epsilon_decay})")
    print(f"Batch size: {batch_size}")
    print(f"Buffer capacity: {buffer_capacity}")
    print(f"Reward perspective: {reward_perspective}")
    print(f"Color strategy: ALTERNATING (White on even episodes, Black on odd)")
    print(f"Training strategy: Store all moves with game outcome")
    print("=" * 70)
    
    # Initialize environment
    print("\nInitializing Chess960 environment...")
    env = Chess960Env(
        max_moves=max_moves,
        stockfish_path=stockfish_path,
        stockfish_depth=stockfish_depth,
        stockfish_time=0.5,
        reward_perspective=reward_perspective
    )
    
    # Initialize agent
    # Note: state_size=72 because we use 9x8 board (8 rows + 1 row for side-to-move)
    print("Initializing DQN agent...")
    agent = DQNAgent(
        state_size=72,  # 9x8 board with side-to-move info
        action_size=4672,  # Maximum possible chess moves
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update_freq=target_update_freq
    )
    
    # Training metrics
    episode_rewards = []
    episode_losses = []
    epsilon_history = []
    recent_rewards = deque(maxlen=100)
    
    # Track rewards by starting color to verify balance
    white_start_rewards = []  # Games where white moved first
    black_start_rewards = []  # Games where black moved first
    white_start_count = 0
    black_start_count = 0
    
    # Training loop
    print("\nStarting training...\n")
    start_time = time.time()
    
    try:
        for episode in range(1, num_episodes + 1):
            # ============================================================
            # CRITICAL FIX: Alternate starting color each episode
            # ============================================================
            starting_color = chess.WHITE if episode % 2 == 0 else chess.BLACK
            state = env.reset(starting_color=starting_color)
            
            # Track which color started this game
            white_started = (starting_color == chess.WHITE)
            if white_started:
                white_start_count += 1
            else:
                black_start_count += 1
            
            episode_reward = 0.0
            episode_loss = []
            done = False
            step = 0
            
            # Track color-specific metrics for this episode
            episode_white_rewards = 0.0
            episode_black_rewards = 0.0
            episode_white_moves = 0
            episode_black_moves = 0
            
            # Store game history for batch storage at game end
            game_history = []  # List of (state, action, next_state, is_white_turn)
            
            # ============================================================
            # Episode Loop - Play one complete game
            # ============================================================
            while not done:
                # Get current player BEFORE making the move
                current_player = env.get_current_player()
                is_white_turn = (current_player == chess.WHITE)
                
                # Select action using epsilon-greedy policy
                legal_moves_count = env.get_legal_moves_count()
                action = agent.select_action(state, legal_moves_count, training=True)
                
                # Execute action in environment
                next_state, reward, done, info = env.step(action)
                
                # Store move in game history (don't add to replay buffer yet)
                game_history.append((state, action, next_state, is_white_turn))
                
                # Train agent on existing experiences from previous games
                if len(agent.replay_buffer) >= min_replay_size and step % train_frequency == 0:
                    loss = agent.train_step()
                    if loss is not None:
                        episode_loss.append(loss)
                
                state = next_state
                step += 1
            
            # ============================================================
            # Game Finished - Store all moves with final outcome
            # ============================================================
            if reward != 0:
                # Store all moves from this game with the final reward
                store_game_outcome(agent.replay_buffer, game_history, reward)
                
                # Track statistics by color
                for _, _, _, is_white in game_history:
                    if is_white:
                        episode_white_rewards += reward
                        episode_white_moves += 1
                    else:
                        # Black's reward is the negation of white's reward
                        episode_black_rewards += (-reward)
                        episode_black_moves += 1
            
            # Track final game outcome by starting color
            episode_reward = reward
            if white_started:
                white_start_rewards.append(reward)
            else:
                black_start_rewards.append(reward)
            
            # Update exploration rate
            agent.decay_epsilon()
            agent.episodes += 1
            
            # ============================================================
            # Record Metrics
            # ============================================================
            episode_rewards.append(episode_reward)
            recent_rewards.append(episode_reward)
            epsilon_history.append(agent.epsilon)
            
            # Calculate average loss
            avg_loss = np.mean(episode_loss) if episode_loss else None
            if avg_loss is not None:
                episode_losses.append(avg_loss)
            
            avg_reward = np.mean(recent_rewards)
            
            # ============================================================
            # Logging
            # ============================================================
            if episode % 10 == 0:
                elapsed_time = time.time() - start_time
                eps_per_min = episode / (elapsed_time / 60)
                
                loss_str = f"{avg_loss:.6f}" if avg_loss is not None else "N/A"
                
                # Calculate average rewards by starting color (last 100 episodes)
                recent_white_start = white_start_rewards[-100:] if len(white_start_rewards) > 0 else [0]
                recent_black_start = black_start_rewards[-100:] if len(black_start_rewards) > 0 else [0]
                avg_white_start = np.mean(recent_white_start)
                avg_black_start = np.mean(recent_black_start)
                
                # Show which color started this episode
                color_indicator = "W" if white_started else "B"
                
                print(f"Episode {episode:4d}/{num_episodes} [{color_indicator}] | "
                      f"Reward: {episode_reward:7.4f} | "
                      f"Avg(100): {avg_reward:7.4f} | "
                      f"WS/BS: {avg_white_start:6.3f}/{avg_black_start:6.3f} | "
                      f"ε: {agent.epsilon:.4f} | "
                      f"Loss: {loss_str:>8} | "
                      f"Speed: {eps_per_min:.1f} ep/min")
            
            # Log to file
            log_training_progress(
                episode=episode,
                reward=episode_reward,
                epsilon=agent.epsilon,
                loss=avg_loss,
                avg_reward=avg_reward,
                log_file=os.path.join(log_dir, "training.log")
            )
            
            # Log color-specific statistics
            if episode % 10 == 0:
                color_log_file = os.path.join(log_dir, "color_stats.log")
                with open(color_log_file, 'a') as f:
                    avg_ws = np.mean(white_start_rewards[-100:]) if white_start_rewards else 0
                    avg_bs = np.mean(black_start_rewards[-100:]) if black_start_rewards else 0
                    balance = 1.0 - abs(avg_ws - avg_bs) if (avg_ws != 0 or avg_bs != 0) else 1.0
                    f.write(f"Episode {episode}: WhiteStart={avg_ws:.6f}, BlackStart={avg_bs:.6f}, "
                           f"Diff={abs(avg_ws - avg_bs):.6f}, Balance={balance:.4f}, "
                           f"WSCount={white_start_count}, BSCount={black_start_count}\n")
            
            # ============================================================
            # Evaluation
            # ============================================================
            if episode % eval_frequency == 0:
                print("\n" + "-" * 70)
                print(f"Evaluating agent at episode {episode}...")
                eval_metrics = evaluate_agent(env, agent, episodes=eval_episodes, verbose=False)
                
                print(f"Evaluation Results:")
                print(f"  Avg Reward: {eval_metrics['avg_reward']:.4f} ± {eval_metrics['std_reward']:.4f}")
                print(f"  Avg Moves:  {eval_metrics['avg_moves']:.2f}")
                
                # Show starting color balance
                if white_start_rewards and black_start_rewards:
                    recent_ws_avg = np.mean(white_start_rewards[-200:])
                    recent_bs_avg = np.mean(black_start_rewards[-200:])
                    balance = 1.0 - abs(recent_ws_avg - recent_bs_avg)
                    
                    print(f"  Starting Color Balance (last 200 games):")
                    print(f"    White-start avg reward: {recent_ws_avg:6.4f}")
                    print(f"    Black-start avg reward: {recent_bs_avg:6.4f}")
                    print(f"    Balance score:          {balance:6.4f} (1.0 = perfect)")
                    print(f"    White-start games: {len([r for r in white_start_rewards[-200:]])} (target: 100)")
                    print(f"    Black-start games: {len([r for r in black_start_rewards[-200:]])} (target: 100)")
                    
                    # Verify alternation is working
                    expected_ws = episode // 2
                    expected_bs = (episode + 1) // 2
                    print(f"  Color Distribution Check:")
                    print(f"    Expected W/B: {expected_ws}/{expected_bs}")
                    print(f"    Actual W/B:   {white_start_count}/{black_start_count}")
                    if white_start_count == expected_ws and black_start_count == expected_bs:
                        print(f" Alternation working perfectly!")
                    else:
                        print(f" Warning: Color counts don't match expected alternation")
                
                print("-" * 70 + "\n")
                
                # Save example game (alternate starting color)
                example_color = chess.WHITE if episode % 100 < 50 else chess.BLACK
                game_file = os.path.join(log_dir, f"example_game_ep{episode}.txt")
                save_example_game(env, agent, game_file, seed=42)
            
            # ============================================================
            # Save Checkpoint
            # ============================================================
            if episode % save_frequency == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_ep{episode}.pth")
                agent.save(checkpoint_path)
        
        # ============================================================
        # Training Complete - Final Evaluation
        # ============================================================
        print("\n" + "=" * 70)
        print("Training completed! Running final evaluation...")
        print("=" * 70)
        
        final_metrics = evaluate_agent(env, agent, episodes=50, verbose=True)
        
        # Save final model
        final_model_path = os.path.join(save_dir, "final_model.pth")
        agent.save(final_model_path)
        
        # Generate plots
        print("\nGenerating training visualizations...")
        plot_training_metrics(
            episode_rewards,
            episode_losses,
            epsilon_history,
            save_path=os.path.join(log_dir, "training_metrics.png")
        )
        
        plot_reward_distribution(
            episode_rewards,
            save_path=os.path.join(log_dir, "reward_distribution.png")
        )
        
        # Save color-specific statistics
        color_stats_file = os.path.join(log_dir, "color_statistics.txt")
        with open(color_stats_file, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("Starting Color Training Statistics\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Game Distribution by Starting Color:\n")
            f.write(f"  White started: {white_start_count} games\n")
            f.write(f"  Black started: {black_start_count} games\n")
            f.write(f"  Ratio: {white_start_count / max(black_start_count, 1):.3f}\n")
            f.write(f"  Expected ratio: ~1.00 (perfect alternation)\n\n")
            
            f.write("Average Rewards by Starting Color:\n")
            f.write(f"  White-start games: {np.mean(white_start_rewards):.6f} ± {np.std(white_start_rewards):.6f}\n")
            f.write(f"  Black-start games: {np.mean(black_start_rewards):.6f} ± {np.std(black_start_rewards):.6f}\n\n")
            
            reward_diff = abs(np.mean(white_start_rewards) - np.mean(black_start_rewards))
            balance_score = 1.0 - reward_diff
            
            f.write("Balance Metrics:\n")
            f.write(f"  Reward difference: {reward_diff:.6f}\n")
            f.write(f"  Balance score: {balance_score:.4f}\n")
            f.write(f"  (1.0 = perfect balance, 0.0 = maximum bias)\n\n")
            
            f.write("Interpretation:\n")
            if balance_score > 0.9:
                f.write(" Excellent balance - no color bias detected\n")
            elif balance_score > 0.7:
                f.write(" Good balance - minor variance is normal\n")
            elif balance_score > 0.5:
                f.write(" Moderate imbalance - monitor further training\n")
            else:
                f.write(" Poor balance - investigate reward calculation\n")
            
            f.write("\nColor Alternation Verification:\n")
            expected_ws = num_episodes // 2
            expected_bs = (num_episodes + 1) // 2
            f.write(f"  Expected white-start: {expected_ws}\n")
            f.write(f"  Actual white-start:   {white_start_count}\n")
            f.write(f"  Expected black-start: {expected_bs}\n")
            f.write(f"  Actual black-start:   {black_start_count}\n")
            if white_start_count == expected_ws and black_start_count == expected_bs:
                f.write(f" Alternation maintained!\n")
            else:
                f.write(f" Alternation deviation detected\n")
        
        print(f"Color statistics saved to: {color_stats_file}")
        
        # Create training summary
        total_time = time.time() - start_time
        create_training_summary(
            total_episodes=num_episodes,
            final_metrics=final_metrics,
            training_time=total_time,
            save_path=os.path.join(log_dir, "training_summary.txt")
        )
        
        # Print final summary
        print("\n" + "=" * 70)
        print("Training Summary")
        print("=" * 70)
        print(f"Total Episodes: {num_episodes}")
        print(f"Total Time: {total_time/3600:.2f} hours")
        print(f"Final Avg Reward: {final_metrics['avg_reward']:.4f}")
        balance = 1.0 - abs(np.mean(white_start_rewards) - np.mean(black_start_rewards))
        print(f"Starting Color Balance: {balance:.4f} (1.0 = perfect)")
        print(f"White-start: {white_start_count} | Black-start: {black_start_count}")
        print(f"Expected ratio: 1.00 | Actual ratio: {white_start_count / max(black_start_count, 1):.3f}")
        print(f"Model saved to: {final_model_path}")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving checkpoint...")
        interrupt_path = os.path.join(save_dir, "interrupted_model.pth")
        agent.save(interrupt_path)
        print(f"Model saved to: {interrupt_path}")
    
    finally:
        # Clean up resources
        print("\nClosing environment...")
        env.close()
        print("Done!")


if __name__ == "__main__":

    # Training hyperparameters
    NUM_EPISODES = 5000
    MAX_MOVES = 25
    EVAL_FREQUENCY = 50
    EVAL_EPISODES = 10
    
    # Agent hyperparameters
    LEARNING_RATE = 0.0003  
    GAMMA = 0.99  # Discount factor for future rewards
    EPSILON_START = 1.0  # Initial exploration rate
    EPSILON_END = 0.02  # Minimum exploration rate 
    EPSILON_DECAY = 0.999  # Exponential decay per episode
    
    # Network architecture
    HIDDEN_SIZES = [1024, 512, 256]  # Deep Q-Network hidden layers
    
    # Replay buffer settings
    BUFFER_CAPACITY = 200000  # Maximum experiences to store
    BATCH_SIZE = 64  # Minibatch size for training
    MIN_REPLAY_SIZE = 5000  # Start training after this many experiences
    
    # Training parameters
    TARGET_UPDATE_FREQ = 1000  # Update target network every N steps
    TRAIN_FREQUENCY = 1  # Train every N steps
    
    # Stockfish settings
    STOCKFISH_PATH = r"D:\stockfish_17.1\stockfish\stockfish-windows-x86-64-avx2.exe"
    STOCKFISH_DEPTH = 15  # Search depth for position evaluation
    
    # Saving and logging
    SAVE_DIR = "checkpoints"
    LOG_DIR = "logs"
    SAVE_FREQUENCY = 500  # Save checkpoint every N episodes
    
    # Reward perspective - "white" for proper self-play training
    REWARD_PERSPECTIVE = "white"

    train(
        num_episodes=NUM_EPISODES,
        max_moves=MAX_MOVES,
        eval_frequency=EVAL_FREQUENCY,
        eval_episodes=EVAL_EPISODES,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        hidden_sizes=HIDDEN_SIZES,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        min_replay_size=MIN_REPLAY_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        train_frequency=TRAIN_FREQUENCY,
        stockfish_path=STOCKFISH_PATH,
        stockfish_depth=STOCKFISH_DEPTH,
        save_dir=SAVE_DIR,
        log_dir=LOG_DIR,
        save_frequency=SAVE_FREQUENCY,
        reward_perspective=REWARD_PERSPECTIVE
    )