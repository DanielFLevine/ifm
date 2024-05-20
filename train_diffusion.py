import argparse
import json
import logging
import math
import os
import pickle
import random
import time
from datetime import datetime
from itertools import cycle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import torch
import wandb
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader

from utils.modules import MLPLayer, MidFC

logging.basicConfig(format='[%(levelname)s:%(asctime)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = "/home/dfl32/project/huggingface"
if "LOCAL_RANK" in os.environ:
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
else:
    LOCAL_RANK = 0


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm_dataset_path",
        type=str,
        default="/home/dfl32/scratch/cinemaot_data/ifm_hf_ds/gaussian_pca768_normFalse_hf_ds"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/dfl32/scratch/training-runs"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
    )
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=100000
    )
    parser.add_argument(
        "--denoising_time_steps",
        type=int,
        default=100
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=768
    )
    parser.add_argument(
        "--mlp_width",
        type=int,
        default=1024
    )
    parser.add_argument(
        "--wandb_log_steps",
        type=int,
        default=100
    )
    parser.add_argument(
        "--eval_dataset_size",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--intermediate_dim",
        type=int,
        default=1024
    )
    parser.add_argument(
        "--num_fc_layers",
        type=int,
        default=2
    )
    return parser.parse_args()


class CustomDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Convert list of floats to tensor
        return torch.tensor(self.dataset[idx]['expr'], dtype=torch.float32)

    
class DiffusionMLP(nn.Module):
    def __init__(self, denoising_time_steps):
        super(DiffusionMLP, self).__init__()
        self.time_embed = nn.Embedding(denoising_time_steps, 1024)  # 100 is the number of timesteps
        self.model = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 768)
        )
    
    def forward(self, x, t):
        t_embed = self.time_embed(t)
        x = self.model[0](x) + t_embed
        x = self.model[1](x)
        for layer in self.model[2:]:
            x = layer(x)
        return x

class DiffusionFC(nn.Module):
    def __init__(self, intermediate_dim=1024, num_fc_layers=2, denoising_time_steps=100):
        super(DiffusionFC, self).__init__()
        self.time_embed = nn.Embedding(denoising_time_steps, intermediate_dim)
        self.model = nn.Sequential(
            nn.Linear(768, intermediate_dim),
            MidFC(dim=intermediate_dim, num_layers=num_fc_layers),
            nn.Linear(intermediate_dim, 768)
        )
    
    def forward(self, x, t):
        t_embed = self.time_embed(t)
        x = self.model[0](x) + t_embed
        for layer in self.model[1:]:
            x = layer(x)
        return x


class DDPM:
    def __init__(self, model, device, num_timesteps=100, beta_start=0.0001, beta_end=0.02):
        self.model = model
        self.device = device
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.num_timesteps, (batch_size,)).to(self.device)

    def forward_diffusion(self, x_0, t):
        alpha_bars_t = self.alpha_bars[t].unsqueeze(1)
        noise = torch.randn_like(x_0).to(self.device)
        x_t = torch.sqrt(alpha_bars_t) * x_0 + torch.sqrt(1 - alpha_bars_t) * noise
        return x_t, noise

    def loss(self, x_0):
        batch_size = x_0.shape[0]
        t = self.sample_timesteps(batch_size).to(x_0.device)
        x_t, noise = self.forward_diffusion(x_0, t)
        noise_pred = self.model(x_t, t)
        return nn.MSELoss()(noise_pred, noise)

def save_model(model, step, run_name, directory="/home/dfl32/scratch/training-runs/simple_ifm/"):
    # Ensure the directory exists
    dir_path = os.path.join(directory, run_name)
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, f"checkpoint-{step}.pt")
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")

def run_eval_loop(ddpm, val_dataloader, device, args):
    ddpm.model.eval()
    losses = []
    for batch in tqdm(val_dataloader):
        x_0 = batch.to(device)
        loss = ddpm.loss(x_0)
        losses.append(loss.item())
    eval_loss = sum(losses)/len(losses)
    ddpm.model.train()
    return eval_loss

def main(args):
    now = datetime.now()
    now = datetime.strftime(now, "%Y-%m-%d_%H-%M-%S")
    run_name = f"diffusion-{now}"
    wandb.init(
            project="IFM",
            name=run_name,
        )
    device = torch.device("cuda")

    model = DiffusionFC(
        intermediate_dim=args.intermediate_dim,
        num_fc_layers=args.num_fc_layers,
        denoising_time_steps=args.denoising_time_steps
    ).to(device)
    ddpm = DDPM(model, device, num_timesteps=args.denoising_time_steps)

    wandb.watch(model, log="all", log_freq=10)

    optimizer = torch.optim.Adam(model.parameters())

    dataset = load_from_disk(args.llm_dataset_path)

    split_dataset = dataset.train_test_split(
        test_size=0.1, 
        shuffle=True, 
        seed=42
    )

    train_dataset = CustomDataset(split_dataset['train'])
    val_dataset = split_dataset['test']
    val_dataset = CustomDataset(val_dataset.select(range(min(len(val_dataset), args.eval_dataset_size))))

    logger.info(train_dataset.dataset)
    logger.info(val_dataset.dataset)

    logger.info("Setting up dataloaders...")

    train_dataloader = cycle(DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True))
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for step in tqdm(range(args.num_train_steps)):
        x_0= next(train_dataloader)
        x_0 = x_0.to(device)
        optimizer.zero_grad()
        loss = ddpm.loss(x_0)
        loss.backward()
        optimizer.step()

        # Log to wandb
        if (step+1) % args.wandb_log_steps == 0:
            eval_loss = run_eval_loop(ddpm, val_dataloader, device, args)
            wandb.log({
                "loss": loss.item(),
                "eval loss": eval_loss
                })
            logger.info(f"Loss = {loss.item()}, Eval Loss = {eval_loss}")

        if (step+1) % args.save_steps == 0:
            save_model(model, step+1, run_name)

# # Denoising function
# def denoise(model, ddpm, x_noisy, num_steps=100):
#     model.eval()
#     x = x_noisy.clone().detach()
#     for t in range(num_steps-1, -1, -1):
#         t_tensor = torch.tensor([t], dtype=torch.long).to(x.device)
#         x_t = ddpm.forward_diffusion(x, t_tensor)[0]
#         x = model(x_t, t_tensor)
#     return x

# # Example denoising
# x_noisy = torch.randn(1, 768)  # Replace with your noisy input
# x_denoised = denoise(model, ddpm, x_noisy)
# print(x_denoised)

if __name__ == "__main__":
    args = parse_arguments()
    logger.info(args)
    main(args)
