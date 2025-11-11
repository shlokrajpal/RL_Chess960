#chess960_env.py

import chess
import chess.engine
import numpy as np
from typing import Tuple, Dict, Optional


class Chess960Env:
    # Piece encoding: 0=empty, 1-6=white pieces, 7-12=black pieces
    PIECE_TO_INT = {
        None: 0,
        chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
        chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6
    }
    
    def __init__(
        self,
        max_moves: int = 30,
        stockfish_path: str = r"D:\stockfish_17.1\stockfish\stockfish-windows-x86-64-avx2.exe",
        stockfish_depth: int = 15,
        stockfish_time: float = 0.1,
        reward_perspective: str = "white"  # "white", "black", or "current"
    ):
        self.max_moves = max_moves
        self.stockfish_path = stockfish_path
        self.stockfish_depth = stockfish_depth
        self.stockfish_time = stockfish_time
        self.reward_perspective = reward_perspective
        
        self.board = None
        self.move_count = 0
        self.engine = None
        
        # Initialize Stockfish engine
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        except Exception as e:
            print(f"Warning: Could not initialize Stockfish: {e}")
            print("Reward calculation will return 0.0")
    
    def reset(
        self, 
        seed: Optional[int] = None,
        starting_color: chess.Color = chess.WHITE
    ) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        
        # Create new Chess960 position
        self.board = chess.Board.from_chess960_pos(
            np.random.randint(0, 960)
        )
        
        # Set which color moves first
        self.board.turn = starting_color
        self.move_count = 0
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        legal_moves = list(self.board.legal_moves)
        
        # Validate action
        if action < 0 or action >= len(legal_moves):
            # Invalid action: return penalty and terminate
            return self._get_state(), -1.0, True, {"error": "invalid_action"}
        
        # Store whose turn it was before the move
        player_color = self.board.turn  # chess.WHITE or chess.BLACK
        
        # Execute move
        move = legal_moves[action]
        self.board.push(move)
        self.move_count += 1
        
        # Check if episode is done
        done = (
            self.move_count >= self.max_moves or
            self.board.is_game_over()
        )
        
        # Calculate reward only at episode end
        reward = 0.0
        if done:
            reward = self._calculate_reward(player_color)
        
        info = {
            "move_count": self.move_count,
            "legal_moves": len(list(self.board.legal_moves)),
            "last_move": move.uci(),
            "player_color": "white" if player_color == chess.WHITE else "black"
        }
        
        return self._get_state(), reward, done, info
    
    def _get_state(self) -> np.ndarray:
        state = np.zeros((9, 8), dtype=np.int8)
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            row = 7 - (square // 8)
            col = square % 8
            
            if piece is not None:
                piece_value = self.PIECE_TO_INT[piece.piece_type]
                if piece.color == chess.BLACK:
                    piece_value += 6
                state[row, col] = piece_value
        
        # Add side-to-move information in the 9th row
        state[8, :] = 1 if self.board.turn == chess.WHITE else 0
        
        return state
    
    def _calculate_reward(self, player_color: chess.Color) -> float:
        if self.engine is None:
            return 0.0
        
        try:
            # Get evaluation
            info = self.engine.analyse(
                self.board,
                chess.engine.Limit(
                    depth=self.stockfish_depth,
                    time=self.stockfish_time
                )
            )
            
            score = info["score"]
            
            # Get score from configured perspective
            if self.reward_perspective == "white":
                eval_score = score.white()
            elif self.reward_perspective == "black":
                eval_score = score.black()
            else:  # "current"
                eval_score = score.relative
            
            # Convert to centipawns
            if eval_score.is_mate():
                mate_in = eval_score.mate()
                eval_cp = 10000 if mate_in > 0 else -10000
            else:
                eval_cp = eval_score.score()
            
            # Scale to [-1, 1] using tanh
            reward = np.tanh(eval_cp / 100.0)
            
            return float(reward)
            
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return 0.0
    
    def get_legal_moves_count(self) -> int:
        return len(list(self.board.legal_moves))
    
    def get_current_player(self) -> chess.Color:
        """
        Returns:
            chess.WHITE or chess.BLACK
        """
        return self.board.turn
    
    def render(self) -> str:
        """
        Returns:
            Unicode string showing the board
        """
        return str(self.board)
    
    def close(self):
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                pass  
            self.engine = None