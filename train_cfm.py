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

import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
import torchdyn
import wandb
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import pairwise
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *

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
        "--train_dataset_size",
        type=int,
        default=None
    )
    parser.add_argument(
        "--eval_dataset_size",
        type=int,
        default=1000
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
        "--input_dim",
        type=int,
        default=768
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.01
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
    return parser.parse_args()


class CustomDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Convert list of floats to tensor
        return torch.tensor(self.dataset[idx]['expr'], dtype=torch.float32)


def sample_conditional_pt(x0, x1, t, sigma):
    """
    Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    xt : Tensor, shape (bs, *dim)

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    t = t.reshape(-1, *([1] * (x0.dim() - 1)))
    mu_t = t * x1 + (1 - t) * x0
    epsilon = torch.randn_like(x0)
    return mu_t + sigma * epsilon

def compute_conditional_vector_field(x0, x1):
    """
    Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch

    Returns
    -------
    ut : conditional vector field ut(x1|x0) = x1 - x0

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    return x1 - x0

def mmd_rbf(X, Y, gamma=2.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = pairwise.rbf_kernel(X, X, gamma)
    YY = pairwise.rbf_kernel(Y, Y, gamma)
    XY = pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def compute_wass(X, Y, reg=0.01):
    # Compute the cost matrix (squared Euclidean distances)
    M = ot.dist(X, Y, metric='sqeuclidean')
    
    # Normalize the cost matrix
    M /= M.max()
    
    # Assume uniform distribution of weights
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]
    
    wasserstein_dist = ot.sinkhorn2(a, b, M, reg)
    return wasserstein_dist

def save_model(model, step, run_name, directory="/home/dfl32/scratch/training-runs/simple_ifm/"):
    # Ensure the directory exists
    dir_path = os.path.join(directory, run_name)
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, f"checkpoint-{step}.pt")
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")

def run_eval_loop(model, val_dataloader, device, args):
    model.eval()
    losses = []
    for batch in tqdm(val_dataloader):
        x1 = batch.to(device)
        x0 = torch.normal(0.0, 1.0**0.5, size=x1.shape).to(device)

        t = torch.rand(x0.shape[0]).type_as(x0)
        xt = sample_conditional_pt(x0, x1, t, sigma=args.sigma)
        ut = compute_conditional_vector_field(x0, x1)

        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        loss = torch.mean((vt - ut) ** 2)
        losses.append(loss.item())
    eval_loss = sum(losses)/len(losses)
    model.train()
    return eval_loss



def main(args):
    now = datetime.now()
    now = datetime.strftime(now, "%Y-%m-%d_%H-%M-%S")
    run_name = f"cfm-mlp-{now}"
    wandb.init(
            project="IFM",
            name=run_name,
        )
    device = torch.device("cuda")

    model = MLP(dim=args.input_dim, w=args.mlp_width, time_varying=True).to(device)
    wandb.watch(model, log="all", log_freq=10)

    optimizer = torch.optim.Adam(model.parameters())

    dataset = load_from_disk(args.llm_dataset_path)
    if args.train_dataset_size:
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(args.train_dataset_size))

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

    for step in tqdm(range(args.num_train_steps)):
        
        optimizer.zero_grad()

        x1 = next(train_dataloader)
        x1 = x1.to(device)

        x0 = torch.normal(0.0, 1.0**0.5, size=x1.shape).to(device)

        t = torch.rand(x0.shape[0]).type_as(x0)
        xt = sample_conditional_pt(x0, x1, t, sigma=args.sigma)
        ut = compute_conditional_vector_field(x0, x1)

        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        loss = torch.mean((vt - ut) ** 2)

        loss.backward()
        optimizer.step()

        # Log to wandb
        if (step+1) % args.wandb_log_steps == 0:
            eval_loss = run_eval_loop(model, val_dataloader, device, args)
            wandb.log({
                "loss": loss.item(),
                "eval loss": eval_loss
                })
            logger.info(f"Loss = {loss.item()}, Eval Loss = {eval_loss}")

        if (step+1) % args.save_steps == 0:
            save_model(model, step+1, run_name)


        # if (k + 1) % 5000 == 0:
        #     end = time.time()
        #     print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
        #     start = end
        #     node = NeuralODE(
        #         torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        #     )
        #     with torch.no_grad():
        #         traj = node.trajectory(
        #             sample_8gaussians(1024),
        #             t_span=torch.linspace(0, 1, 100),
        #         )
        #         plot_trajectories(traj.cpu().numpy())

if __name__ == "__main__":
    args = parse_arguments()
    logger.info(args)
    main(args)