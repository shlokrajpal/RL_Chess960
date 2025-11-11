#evaluate.py

import os
import sys
import time
import chess
import chess.engine
import chess.pgn
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.chess960_env import Chess960Env
from src.rl.dqn_agent import DQNAgent


class Chess960Evaluator:
    def __init__(
        self,
        agent: DQNAgent,
        stockfish_path: str,
        max_moves: int = 30,
        stockfish_time: float = 0.5,
        stockfish_depth: int = 15
    ):
        self.agent = agent
        self.max_moves = max_moves
        self.stockfish_time = stockfish_time
        self.stockfish_depth = stockfish_depth
        
        # Initialize Stockfish
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Stockfish: {e}")
    
    def get_stockfish_move(self, board: chess.Board) -> chess.Move:
        result = self.engine.play(
            board,
            chess.engine.Limit(
                time=self.stockfish_time,
                depth=self.stockfish_depth
            )
        )
        return result.move
    
    def get_agent_move(self, board: chess.Board) -> chess.Move:
        # Convert board to state representation (matching training format)
        state = self._board_to_state(board)
        
        # Get legal moves count
        legal_moves = list(board.legal_moves)
        legal_moves_count = len(legal_moves)
        
        # Select action (no exploration)
        action = self.agent.select_action(state, legal_moves_count, training=False)
        
        # Convert action to move
        return legal_moves[action]
    
    def _board_to_state(self, board: chess.Board) -> np.ndarray:
        state = np.zeros((9, 8), dtype=np.int8)
        
        piece_to_int = {
            chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
            chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6
        }
        
        # Encode board position
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            row = 7 - (square // 8)
            col = square % 8
            
            if piece is not None:
                piece_value = piece_to_int[piece.piece_type]
                if piece.color == chess.BLACK:
                    piece_value += 6
                state[row, col] = piece_value
        
        # Add side-to-move information in the 9th row (same as training)
        state[8, :] = 1 if board.turn == chess.WHITE else 0
        
        return state
    
    def get_position_evaluation(self, board: chess.Board) -> float:
        info = self.engine.analyse(
            board,
            chess.engine.Limit(depth=self.stockfish_depth, time=0.1)
        )
        
        score = info["score"].white()
        
        if score.is_mate():
            mate_in = score.mate()
            return 100.0 if mate_in > 0 else -100.0
        else:
            return score.score() / 100.0  # Convert centipawns to pawns
    
    def play_game(
        self,
        agent_color: chess.Color,
        starting_position: int = None,
        verbose: bool = False
    ) -> Dict:
        # Initialize board
        if starting_position is None:
            starting_position = np.random.randint(0, 960)
        
        board = chess.Board.from_chess960_pos(starting_position)
        
        if verbose:
            print(f"\nStarting position {starting_position}")
            print(f"Agent plays as: {'White' if agent_color == chess.WHITE else 'Black'}")
            print(board)
            print()
        
        # Track game data
        moves = []
        evaluations = []
        move_times = []
        
        initial_eval = self.get_position_evaluation(board)
        evaluations.append(initial_eval)
        
        # Play game
        for move_num in range(self.max_moves):
            if board.is_game_over():
                break
            
            move_start = time.time()
            
            # Determine whose turn
            if board.turn == agent_color:
                # Agent's turn
                move = self.get_agent_move(board)
                player = "Agent"
            else:
                # Stockfish's turn
                move = self.get_stockfish_move(board)
                player = "Stockfish"
            
            move_time = time.time() - move_start
            move_times.append(move_time)
            
            # Execute move
            moves.append(move)
            board.push(move)
            
            # Evaluate position
            eval_score = self.get_position_evaluation(board)
            evaluations.append(eval_score)
            
            if verbose:
                turn = "White" if board.turn == chess.BLACK else "Black"  # After move
                print(f"Move {move_num + 1} ({turn} - {player}): {move.uci()} "
                      f"[Eval: {eval_score:+.2f}] [{move_time:.3f}s]")
        
        # Calculate final evaluation and reward
        final_eval = evaluations[-1]
        
        # Reward from agent's perspective
        if agent_color == chess.WHITE:
            reward = np.tanh(final_eval / 10.0)  # Positive = good for agent
        else:
            reward = np.tanh(-final_eval / 10.0)  # Negative eval = good for agent
        
        # Determine outcome
        eval_threshold = 2.0  # Pawns
        if agent_color == chess.WHITE:
            if final_eval > eval_threshold:
                outcome = "win"
            elif final_eval < -eval_threshold:
                outcome = "loss"
            else:
                outcome = "draw"
        else:
            if final_eval < -eval_threshold:
                outcome = "win"
            elif final_eval > eval_threshold:
                outcome = "loss"
            else:
                outcome = "draw"
        
        if verbose:
            print(f"\nGame ended after {len(moves)} moves")
            print(f"Final evaluation: {final_eval:+.2f}")
            print(f"Agent reward: {reward:.4f}")
            print(f"Outcome: {outcome}")
        
        return {
            "starting_position": starting_position,
            "agent_color": "white" if agent_color == chess.WHITE else "black",
            "moves": moves,
            "evaluations": evaluations,
            "move_times": move_times,
            "final_eval": final_eval,
            "reward": reward,
            "outcome": outcome,
            "num_moves": len(moves),
            "board": board
        }
    
    def save_game_pgn(self, game_data: Dict, filepath: str):
        """
        Save game in PGN format for analysis.
        """
        game = chess.pgn.Game()
        
        # Set headers
        game.headers["Event"] = "Chess960 Agent Evaluation"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["White"] = "Agent" if game_data["agent_color"] == "white" else "Stockfish"
        game.headers["Black"] = "Stockfish" if game_data["agent_color"] == "white" else "Agent"
        game.headers["Variant"] = "Chess960"
        game.headers["FEN"] = chess.Board.from_chess960_pos(
            game_data["starting_position"]
        ).fen()
        game.headers["Result"] = "*"  # Incomplete game (opening only)
        
        # Add moves
        node = game
        board = chess.Board.from_chess960_pos(game_data["starting_position"])
        
        for i, move in enumerate(game_data["moves"]):
            node = node.add_variation(move)
            board.push(move)
            
            # Add evaluation as comment
            if i < len(game_data["evaluations"]) - 1:
                eval_score = game_data["evaluations"][i + 1]
                node.comment = f"[{eval_score:+.2f}]"
        
        # Save to file
        with open(filepath, "w") as f:
            f.write(str(game))
    
    def close(self):
        """Close Stockfish engine."""
        if self.engine:
            self.engine.quit()


if __name__ == "__main__":

    CHECKPOINT_PATH = "checkpoints/final_model.pth"
    STATE_SIZE = 72  
    ACTION_SIZE = 4672
    HIDDEN_SIZES = [1024, 512, 256]
    
    # Evaluation Configuration
    NUM_GAMES = 100              # Total number of games to play
    MAX_MOVES = 25               # Maximum moves per game 
    SAVE_GAMES = False            # Whether to save PGN files
    NUM_GAMES_TO_SAVE = 50       # Number of games to save as PGN
    
    # Stockfish Configuration
    #STOCKFISH_PATH = r"D:\stockfish_7\stockfish-7-win\Windows\stockfish 7 x64.exe"
    STOCKFISH_PATH = r"D:\stockfish_17.1\stockfish\stockfish-windows-x86-64-avx2.exe"
    STOCKFISH_TIME = 0.1         # Time limit per Stockfish move (seconds)
    STOCKFISH_DEPTH = 10         # Search depth for Stockfish
    
    # Output Configuration
    OUTPUT_DIR = "evaluation_results"
    
    # Evaluation Thresholds
    EVAL_THRESHOLD = 2.0         # difference for win/loss determination

    print("=" * 70)
    print("Chess960 Agent Evaluation vs Stockfish")
    print("=" * 70)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"State size: {STATE_SIZE} (9x8 board with side-to-move)")
    print(f"Games: {NUM_GAMES} ({NUM_GAMES//2} as White, {NUM_GAMES//2} as Black)")
    print(f"Max moves per game: {MAX_MOVES}")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if SAVE_GAMES:
        games_dir = os.path.join(OUTPUT_DIR, "games")
        os.makedirs(games_dir, exist_ok=True)
    
    # Load agent
    print("\nLoading agent...")
    agent = DQNAgent(
        state_size=STATE_SIZE,  
        action_size=ACTION_SIZE,
        hidden_sizes=HIDDEN_SIZES
    )
    agent.load(CHECKPOINT_PATH)
    agent.epsilon = 0.0  # No exploration during evaluation
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = Chess960Evaluator(
        agent=agent,
        stockfish_path=STOCKFISH_PATH,
        max_moves=MAX_MOVES,
        stockfish_time=STOCKFISH_TIME,
        stockfish_depth=STOCKFISH_DEPTH
    )
    
    # Play games
    print(f"\nPlaying {NUM_GAMES} games...\n")
    
    results = {
        "white": {"games": [], "win": 0, "draw": 0, "loss": 0},
        "black": {"games": [], "win": 0, "draw": 0, "loss": 0}
    }
    
    all_rewards = []
    all_final_evals = []
    all_move_counts = []
    
    start_time = time.time()
    
    for game_num in range(NUM_GAMES):
        # Alternate colors
        agent_color = chess.WHITE if game_num % 2 == 0 else chess.BLACK
        color_str = "white" if agent_color == chess.WHITE else "black"
        
        print(f"Game {game_num + 1}/{NUM_GAMES} - Agent as {color_str.capitalize()}...", end=" ")
        
        # Play game
        game_data = evaluator.play_game(
            agent_color=agent_color,
            starting_position=None,  # Random position
            verbose=False
        )
        
        # Record results
        results[color_str]["games"].append(game_data)
        results[color_str][game_data["outcome"]] += 1
        
        all_rewards.append(game_data["reward"])
        all_final_evals.append(game_data["final_eval"])
        all_move_counts.append(game_data["num_moves"])
        
        print(f"{game_data['outcome'].upper()} [Reward: {game_data['reward']:+.4f}, "
              f"Eval: {game_data['final_eval']:+.2f}, Moves: {game_data['num_moves']}]")
        
        # Save game PGN
        if SAVE_GAMES and game_num < NUM_GAMES_TO_SAVE:
            pgn_file = os.path.join(
                games_dir,
                f"game_{game_num + 1}_{color_str}_{game_data['outcome']}.pgn"
            )
            evaluator.save_game_pgn(game_data, pgn_file)
    
    elapsed_time = time.time() - start_time
    
    # Calculate statistics
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    # Overall statistics
    total_wins = results["white"]["win"] + results["black"]["win"]
    total_draws = results["white"]["draw"] + results["black"]["draw"]
    total_losses = results["white"]["loss"] + results["black"]["loss"]
    
    win_rate = total_wins / NUM_GAMES * 100
    draw_rate = total_draws / NUM_GAMES * 100
    loss_rate = total_losses / NUM_GAMES * 100
    
    print(f"\nOverall Performance:")
    print(f"  Games Played:     {NUM_GAMES}")
    print(f"  Wins:             {total_wins:3d} ({win_rate:.1f}%)")
    print(f"  Draws:            {total_draws:3d} ({draw_rate:.1f}%)")
    print(f"  Losses:           {total_losses:3d} ({loss_rate:.1f}%)")
    print(f"  Win Rate:         {win_rate:.1f}%")
    
    # Performance by color
    print(f"\nPerformance as White:")
    white_total = NUM_GAMES // 2
    print(f"  Wins:   {results['white']['win']:3d} / {white_total} ({results['white']['win']/white_total*100:.1f}%)")
    print(f"  Draws:  {results['white']['draw']:3d} / {white_total} ({results['white']['draw']/white_total*100:.1f}%)")
    print(f"  Losses: {results['white']['loss']:3d} / {white_total} ({results['white']['loss']/white_total*100:.1f}%)")
    
    print(f"\nPerformance as Black:")
    black_total = NUM_GAMES // 2
    print(f"  Wins:   {results['black']['win']:3d} / {black_total} ({results['black']['win']/black_total*100:.1f}%)")
    print(f"  Draws:  {results['black']['draw']:3d} / {black_total} ({results['black']['draw']/black_total*100:.1f}%)")
    print(f"  Losses: {results['black']['loss']:3d} / {black_total} ({results['black']['loss']/black_total*100:.1f}%)")
    
    # Reward statistics
    print(f"\nReward Statistics:")
    print(f"  Average Reward:   {np.mean(all_rewards):+.4f} ± {np.std(all_rewards):.4f}")
    print(f"  Median Reward:    {np.median(all_rewards):+.4f}")
    print(f"  Min Reward:       {np.min(all_rewards):+.4f}")
    print(f"  Max Reward:       {np.max(all_rewards):+.4f}")
    
    # Evaluation statistics
    print(f"\nPosition Evaluation (Final):")
    print(f"  Average Eval:     {np.mean(all_final_evals):+.2f} pawns")
    print(f"  Std Dev:          {np.std(all_final_evals):.2f} pawns")
    
    # Game length statistics
    print(f"\nGame Length:")
    print(f"  Average Moves:    {np.mean(all_move_counts):.1f}")
    print(f"  Min Moves:        {np.min(all_move_counts)}")
    print(f"  Max Moves:        {np.max(all_move_counts)}")
    
    # Time statistics
    print(f"\nTime Statistics:")
    print(f"  Total Time:       {elapsed_time:.1f} seconds")
    print(f"  Time per Game:    {elapsed_time/NUM_GAMES:.1f} seconds")
    
    print("\n" + "=" * 70)
    
    # Save detailed results
    results_file = os.path.join(OUTPUT_DIR, "evaluation_summary.txt")
    with open(results_file, "w") as f:
        f.write("Chess960 Agent Evaluation Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
        f.write(f"State Size: {STATE_SIZE}\n")
        f.write(f"Games: {NUM_GAMES}\n")
        f.write(f"Max Moves: {MAX_MOVES}\n\n")
        
        f.write("Overall Performance:\n")
        f.write(f"  Win Rate:  {win_rate:.1f}%\n")
        f.write(f"  Draw Rate: {draw_rate:.1f}%\n")
        f.write(f"  Loss Rate: {loss_rate:.1f}%\n\n")
        
        f.write("Performance by Color:\n")
        f.write(f"  White: W{results['white']['win']}-D{results['white']['draw']}-L{results['white']['loss']}\n")
        f.write(f"  Black: W{results['black']['win']}-D{results['black']['draw']}-L{results['black']['loss']}\n\n")
        
        f.write("Reward Statistics:\n")
        f.write(f"  Mean:   {np.mean(all_rewards):+.4f}\n")
        f.write(f"  Median: {np.median(all_rewards):+.4f}\n")
        f.write(f"  Std:    {np.std(all_rewards):.4f}\n\n")
        
        f.write("Evaluation Statistics:\n")
        f.write(f"  Mean Final Eval: {np.mean(all_final_evals):+.2f} pawns\n")
        f.write(f"  Std Dev:         {np.std(all_final_evals):.2f} pawns\n")
    
    print(f"\nResults saved to: {results_file}")
    if SAVE_GAMES:
        print(f"Game PGNs saved to: {games_dir}")
    
    # Cleanup
    evaluator.close()
    print("\nEvaluation complete!")