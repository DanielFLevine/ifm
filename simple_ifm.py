import argparse
import logging
import os
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal, Beta
from torch.nn import TransformerEncoder, TransformerEncoderLayer

logging.basicConfig(format='[%(levelname)s:%(asctime)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_steps",
        type=int,
        default=10000
    )
    parser.add_argument(
        "--wandb_log_steps",
        type=int,
        default=100
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=2
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=32
    )
    parser.add_argument(
        "--feedforward_dim",
        type=int,
        default=32
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=4
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64
    )
    parser.add_argument(
        "--space_dim",
        type=int,
        default=1
    )
    return parser.parse_args()

class CustomTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, feedforward_dim, n_heads, space_dim=1):
        super(CustomTransformer, self).__init__()

        # Embedding layer: 2-layer feedforward network with LayerNorm and Dropout
        self.space_dim = space_dim
        self.total_hid_dim = hidden_dim*space_dim
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, self.total_hid_dim),
            nn.LayerNorm(self.total_hid_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.total_hid_dim, self.total_hid_dim),
            nn.LayerNorm(self.total_hid_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # Transformer block configuration
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            activation='relu',
            dropout=0.1
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=1)
        
        # Output MLP with LayerNorm and Dropout
        self.output_mlp = nn.Sequential(
            nn.Linear(self.total_hid_dim, self.total_hid_dim),
            nn.LayerNorm(self.total_hid_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.total_hid_dim, input_dim),
            # nn.LayerNorm(input_dim),
            # nn.ReLU(),
            # nn.Dropout(0.1)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # (seq_len, batch_size, feature)

        seq_len, batch_size, feature = x.shape
        x_reshaped = x.view(seq_len, batch_size, self.space_dim, feature // self.space_dim)
        x_reshaped = x_reshaped.permute(0, 2, 1, 3).contiguous()
        x = x_reshaped.view(seq_len * self.space_dim, batch_size, feature // self.space_dim)

        position_indices = torch.arange(x.shape[0], device=x.device).unsqueeze(1).repeat(1, x.shape[1])
        pos_emb = self.positional_embedding(position_indices)
        x = x + pos_emb

        seq_len = x.shape[0]
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(device)

        x = self.transformer_encoder(x, mask=mask)  # Apply causal mask here
        x = x.squeeze(1)
        x = self.output_mlp(x)
        return x

def generate_batch(batch_size, device):
    normal_dist = Normal(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
    beta_dist = Beta(torch.tensor([2.0]), torch.tensor([5.0]))
    samples = torch.zeros(batch_size, 2, 2)
    samples[:, 0, :] = normal_dist.sample((batch_size,)).to(device)
    samples[:, 1, :] = beta_dist.sample((batch_size,)).unsqueeze(-1).repeat(1, 2).to(device)  # Assuming isotropic
    return samples

def save_model(model, step, run_name, directory="/home/dfl32/scratch/training-runs/simple_ifm/"):
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, run_name, f"checkpoint-{step}.pt")
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    model = CustomTransformer(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        feedforward_dim=args.feedforward_dim,
        n_heads=args.n_heads,
        space_dim=args.space_dim
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    now = datetime.now()
    now = datetime.strftime(now, "%Y-%m-%d_%H-%M-%S")
    run_name = f"gaussian-to-beta-{args.input_dim}d-hd{args.hidden_dim}-ffd{args.feedforward_dim}-nheads{args.n_heads}-{now}"
    wandb.init(
            project="IFM",
            name=run_name,
        )
    wandb.watch(model, log="all", log_freq=10)

    for step in tqdm(range(args.train_steps)):
        model.train()
        optimizer.zero_grad()

        # Generate batch
        inputs = generate_batch(args.batch_size, device)
        targets = inputs.clone()  # Assuming the target is to reconstruct the input

        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Log to wandb
        if step % args.wandb_log_steps == 0:
            wandb.log({"loss": loss.item()})
            logger.info(f"Step {step}: Loss = {loss.item()}")

        if step % args.save_steps == 0:
            save_model(model, step, run_name)

if __name__ == "__main__":
    args = parse_arguments()
    logger.info(args)
    train_model(args)