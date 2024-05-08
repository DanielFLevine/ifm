import torch
from torch import nn


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.1, num_layers=3):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim, elementwise_affine=False),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None
        )
        if num_layers > 2:
            self.middle = nn.Sequential(
                (nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim, elementwise_affine=False),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None
                ) for i in range(num_layers-2))
            )
        self.output = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim, elementwise_affine=False),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
        )
    
    def forward(self, input):
        if input.dim() == 3:
            
