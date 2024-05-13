import torch
from torch import nn
from torch.distributions import Normal


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, dropout_prob=0.1):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        identity = x
        out = self.linear(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        # out += identity  # Skip connection
        return out

class CustomDecoder(nn.Module):
    def __init__(self, hidden_size, input_dim, device, num_blocks=1):
        super(CustomDecoder, self).__init__()
        self.intermediate_dim = max(hidden_size, input_dim)
        self.initial_layer = nn.Sequential(
            nn.Linear(hidden_size, self.intermediate_dim),
            nn.LayerNorm(self.intermediate_dim, elementwise_affine=False),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(self.intermediate_dim) for _ in range(num_blocks)]
        )
        self.final_layer = nn.Linear(self.intermediate_dim, input_dim)
        self.to(device)

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.residual_blocks(x)
        x = self.final_layer(x)
        return x

class CustomVAEDecoder(nn.Module):
    def __init__(self, hidden_size, input_dim, device, num_blocks=1):
        super(CustomVAEDecoder, self).__init__()
        self.mean_encoder = ResidualBlock(hidden_size).to(device)
        self.var_encoder = ResidualBlock(hidden_size).to(device)
        self.var_activation = torch.exp
        self.var_eps = 0.0001
        self.decoder = CustomDecoder(
            hidden_size=hidden_size,
            input_dim=input_dim,
            device=device,
            num_blocks=num_blocks
        )
        self.to(device)

    def forward(self, x):
        mu = self.mean_encoder(x)
        var = self.var_encoder(x)
        var = self.var_activation(var) + self.var_eps
        dist = Normal(mu, var.sqrt())
        latents = dist.rsample()
        outputs = self.decoder(latents)
        return outputs, latents, dist
