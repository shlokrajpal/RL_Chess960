# Chess960 Deep Q-Learning Agent

A reinforcement learning project that trains a Deep Q-Network (DQN) agent to play Chess960 (Fischer Random Chess) using self-play and Stockfish-based position evaluation.

### Environment (`chess960_env.py`)
- Chess960 position generation (960 possible starting positions)
- Stockfish integration for position evaluation
- Configurable move limits and evaluation depth
- Support for different reward perspectives (white/black/current)
- Alternating starting colors for balanced training

### Agent (`dqn_agent.py`)
- Deep Q-Network with customizable architecture
- Experience replay buffer for decorrelated training
- Target network updated periodically
- Epsilon-greedy exploration with decay
- Gradient clipping for training stability

### Training (`train.py`)
- Self-play with color alternation each episode
- Batch experience storage at game end
- Comprehensive logging and checkpointing
- Real-time training metrics
- Color balance tracking

### Evaluation (`evaluate.py`)
- Play games against Stockfish
- Generate PGN files for game analysis
- Detailed performance statistics
- Win/draw/loss tracking by color

##  Code Documentation

### `chess960_env.py`

**Purpose:** Implements the Chess960 game environment as a reinforcement learning environment.

**Key Methods:**
- `reset(seed, starting_color)`: Initialize new game with specified starting color
- `step(action)`: Execute move and return (state, reward, done, info)
- `_get_state()`: Encode board as 9×8 array (8 rows + side-to-move row)
- `_calculate_reward(player_color)`: Get Stockfish evaluation as reward
- `get_legal_moves_count()`: Return number of legal moves
- `get_current_player()`: Return whose turn it is (WHITE/BLACK)

**State Representation:**
- 9×8 integer array
- Rows 0-7: Board position (0=empty, 1-6=white pieces, 7-12=black pieces)
- Row 8: Side to move (1=white, 0=black)

**Reward System:**
- Calculated only at episode end
- Stockfish evaluation scaled by tanh(centipawns/100)
- Range: [-1, 1] where +1 = winning, -1 = losing

### `dqn_agent.py`

**Purpose:** Implements the DQN agent with experience replay.

**Key Components:**

**ReplayBuffer:**
- Stores experiences as (state, action, reward, next_state, done) tuples
- Circular buffer with configurable capacity
- Random sampling for training

**DQNAgent:**
- `select_action(state, legal_moves_count, training)`: Choose action (epsilon-greedy if training)
- `store_experience(...)`: Add experience to replay buffer
- `train_step()`: Sample batch and perform gradient descent
- `decay_epsilon()`: Reduce exploration rate
- `save(filepath)`: Save checkpoint
- `load(filepath)`: Load checkpoint

**Training Algorithm:**
1. Sample batch from replay buffer
2. Compute Q-values from main network
3. Compute target Q-values from target network
4. Calculate Huber loss
5. Backpropagate and update main network
6. Periodically copy weights to target network

### `network.py`

**Purpose:** Defines neural network architectures.

**QNetwork:**
- Fully connected feedforward network
- Configurable hidden layers (default: [1024, 512, 256])
- ReLU activations with dropout
- Xavier initialization
- Outputs Q-values for all possible moves (4672 actions)

### `eval_utils.py`

**Purpose:** Evaluation and visualization utilities.

**Functions:**
- `evaluate_agent(env, agent, episodes)`: Run evaluation games
- `save_example_game(env, agent, filepath)`: Save game as text
- `plot_training_metrics(...)`: Generate training plots
- `plot_reward_distribution(...)`: Plot reward histogram
- `log_training_progress(...)`: Append to training log
- `create_training_summary(...)`: Generate summary report

### `train.py`

**Purpose:** Main training loop with self-play.

**Key Features:**
- **Color Alternation:** Alternates starting color each episode (even=white, odd=black)
- **Batch Storage:** Stores all moves with game outcome at episode end
- **Balanced Training:** Tracks and logs color-specific statistics
- **Checkpointing:** Saves model every 500 episodes
- **Evaluation:** Tests agent every 50 episodes

**Training Flow:**
1. Alternate starting color (white on even episodes, black on odd)
2. Play game until max moves or termination
3. Store all moves with final reward in replay buffer
4. Train on experiences from previous games
5. Log metrics and color balance
6. Periodically evaluate and save checkpoints

### `evaluate.py`

**Purpose:** Comprehensive evaluation against Stockfish.

**Chess960Evaluator Class:**
- `play_game(agent_color, starting_position)`: Play single game
- `get_stockfish_move(board)`: Get Stockfish's move
- `get_agent_move(board)`: Get agent's move
- `get_position_evaluation(board)`: Get Stockfish evaluation
- `save_game_pgn(game_data, filepath)`: Export game as PGN

**Evaluation Metrics:**
- Win/draw/loss rates overall and by color
- Average rewards and position evaluations
- Game length statistics
- Time per game

## Training Details

### State Representation
- **Dimensions:** 9×8 (72 features when flattened)
- **Encoding:** Integer encoding (0-12) + side-to-move indicator
- **Pieces:** 0=empty, 1-6=white pieces (P,N,B,R,Q,K), 7-12=black pieces

### Action Space
- **Size:** 4672 possible chess moves
- **Encoding:** Index into list of legal moves
- **Filtering:** Only legal moves considered during action selection

### Reward Signal
- **Source:** Stockfish position evaluation
- **Calculation:** tanh(centipawns / 100)
- **Range:** [-1, 1]
- **Timing:** Only calculated at episode end
- **Perspective:** Always from white's view (stored with correct sign per player)

### Training Strategy
- **Self-play:** Agent plays both sides
- **Color balance:** Alternates starting color each episode
- **Experience storage:** All moves stored with game outcome at episode end
- **Replay buffer:** Samples from past games for training
- **Target network:** Updated every 1000 steps

### Hyperparameters (Defaults)
```python
LEARNING_RATE = 0.0003
GAMMA = 0.99              # Discount factor
EPSILON_START = 1.0       # Initial exploration
EPSILON_END = 0.02        # Minimum exploration
EPSILON_DECAY = 0.999     # Per-episode decay
BATCH_SIZE = 64
BUFFER_CAPACITY = 200000
TARGET_UPDATE_FREQ = 1000
```
