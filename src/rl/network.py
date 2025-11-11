#network.py

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(
        self,
        input_size: int = 72,
        hidden_sizes: list = [512, 256, 128],
        output_size: int = 4672,
        dropout_rate: float = 0.1
    ):
        super(QNetwork, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer (no activation - raw Q-values)
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Ensure input is flattened
        if state.dim() == 3:  # (batch, rows, cols) 
            state = state.view(state.size(0), -1)
        elif state.dim() == 2:  # (batch, features)
            # Check if it matches expected input size
            if state.size(1) != self.input_size:
                raise ValueError(
                    f"Expected input size {self.input_size}, got {state.size(1)}. "
                    f"Make sure state_size in DQNAgent matches the flattened state dimension."
                )
        else:
            raise ValueError(f"Unexpected input shape: {state.shape}")
        
        return self.network(state)