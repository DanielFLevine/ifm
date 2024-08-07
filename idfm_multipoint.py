import argparse
import logging
import os
from datetime import datetime
from tqdm import tqdm

import numpy as np
import seaborn as sns
import torch
import umap
import wandb
import matplotlib.pyplot as plt
from torch import nn
from torch.nn.functional import mse_loss, normalize
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from utils.toy_datasets import IFMdatasets
from utils.modules import TwoLayerMLP, TwoLayerDecoder

logging.basicConfig(format='[%(levelname)s:%(asctime)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class IDFMModel(nn.Module):

    def __init__(self, encoder, transformer, decoder):
        super().__init__()
        self.encoder = encoder
        self.transformer = transformer
        self.decoder = decoder

    def forward(self, inputs, output_attentions=False):
        embs = self.encoder(inputs)
        hid_states = self.transformer.gpt_neox(inputs_embeds=embs, output_attentions=output_attentions)
        outputs = self.decoder(hid_states.last_hidden_state+embs)
        if output_attentions:
            return outputs, hid_states.attentions, hid_states.last_hidden_state
        return outputs, None, None

class torch_wrapper(nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timepoints",
        type=int,
        default=16
    )
    parser.add_argument(
        "--hdim",
        type=int,
        default=64
    )
    parser.add_argument(
        "--nlayer",
        type=int,
        default=2
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=2
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=40000
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096
    )
    parser.add_argument(
        "--inf_batch_size",
        type=int,
        default=4096
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
        "--lr",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5
    )
    parser.add_argument(
        "--min_dt",
        type=float,
        default=1e-4
    )
    parser.add_argument(
        "--use_rk",
        action="store_true",
        help="Use Runge-Kutta method if this flag is set"
    )
    parser.add_argument(
        "--ada",
        action="store_true",
        help="Use adaptive Runge-Kutta method if this flag is set"
    )
    parser.add_argument(
        "--idfm",
        action="store_true",
        help="Use idfm method if this flag is set"
    )
    parser.add_argument(
        "--continuous_time",
        action="store_true",
        help="Add continuous time coordinate to input"
    )
    parser.add_argument(
        "--output_attentions",
        action="store_true",
        help="Plot attention matrices"
    )
    parser.add_argument(
        "--num_inf_trajs",
        type=int,
        default=1024
    )
    parser.add_argument(
        "--attn_dropout",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--orth_reg",
        action="store_true",
        help="Use orthogonality regularizer"
    )
    parser.add_argument(
        "--or_weight",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--ve",
        action="store_true",
        help="Use variance exploding paths"
    )
    parser.add_argument(
        "--sigma_max",
        type=float,
        default=1e1
    )
    parser.add_argument(
        "--sigma_min",
        type=float,
        default=1e-4
    )
    parser.add_argument(
        "--path_sigma",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--use_last",
        action="store_true",
        help="Use last sampled point in conditional paths"
    )
    parser.add_argument(
        "--linear_sched",
        action="store_true",
        help="Use linear variance schedule for VE paths"
    )
    return parser.parse_args()

def generate_straight_paths(X, Y, device='cuda', time_points=16, path_sigma=0.0):
    W = torch.zeros(X.shape[0], time_points, X.shape[1]).to(device)

    # Define the time points, evenly spaced from 0 to 1, inclusive
    times = torch.linspace(0, 1, time_points)

    # Generate W where each slice W[:, i, :] is based on the linear combination of X and Y
    for i, t in enumerate(times):
        if path_sigma > 0.0:
            variance = path_sigma**2
            # Calculate the convex combination of X and Y
            mean = (1 - t) * X + t * Y

            # Sample from a Gaussian with the calculated mean and variance
            W[:, i, :] = mean + torch.randn_like(X).to(device) * variance
        else:
            W[:, i, :] = (((1 - t) * X) + (t * Y)).to(device)
    
    W = W.to(dtype=torch.float32)
    labels = (Y.unsqueeze(1) - X.unsqueeze(1)).repeat(1,time_points,1)
    return W, labels

def generate_VE_paths(X, Y, device='cuda', time_points=16, sigma_min=1e-4, sigma_max=1e1, use_last=False, linear_sched=False):
    sigma_min = torch.tensor(sigma_min)
    sigma_max = torch.tensor(sigma_max)

    W = torch.zeros(X.shape[0], time_points, X.shape[1]).to(device)

    # Define the time points, evenly spaced from 0 to 1, inclusive
    times = torch.linspace(0, 1, time_points).to(device)

    labels = (Y.unsqueeze(1) - X.unsqueeze(1)).repeat(1,time_points,1)

    cur_start = X
    # Loop over each time point
    for i, t in enumerate(times):
        # Calculate the variance for the current time point
        if linear_sched:
            variance = (sigma_max * (1-t)) + (t * sigma_min)
        else:
            variance = sigma_min * (sigma_max / sigma_min) ** (1-t)

        # Calculate the convex combination of X and Y
        mean = (1 - t) * cur_start + t * Y

        # Sample from a Gaussian with the calculated mean and variance
        W[:, i, :] = mean + torch.randn_like(X).to(device) * variance

        if use_last:
            cur_start = W[:, i, :]
            labels[:, i, :] = Y - cur_start

        log_term = torch.log(sigma_max / sigma_min)
        derivative = -sigma_max * log_term * (sigma_max / sigma_min) ** (-t)
        
        # Divide the derivative by the original variance
        ve_scale = derivative / variance

        labels[:, i, :] = labels[:, i, :]

    return W, labels

def generate_idfm(model, inputs, batch_size, time_points=16, continuous_time=False, device='cuda', output_attentions=False):
    """Generate using Eulers method
    """
    dt = 1/(time_points-1)
    if continuous_time:
        t = torch.linspace(0.0, 1.0, time_points).view(1, time_points, 1).expand(batch_size, time_points, 1).to(device)
    for i in range(time_points-1):
        if continuous_time:
            inputs = torch.cat([inputs, t[:, :i+1]], dim=-1)
        outputs, attentions, last_hidden_states = model(inputs, output_attentions=output_attentions)
        new_output = inputs[:, -1, :-1] + outputs[:, -1]*dt
        inputs = torch.cat([inputs[:, :, :-1], new_output.unsqueeze(1)], dim=1)
    if output_attentions:
        return inputs, attentions, last_hidden_states
    return inputs, None, None

def generate_idfm_2(model, inputs, time_points=16):
    """Generate using 2nd order Runge-Kutta (Midpoint) method"""
    dt = 1 / (time_points - 1)
    while inputs.shape[1] < time_points:
        # Get the last input and its corresponding model output
        last_input = inputs[:, -1]  # shape: (batch_size,)
        output = model(inputs)[:, -1]  # shape: (batch_size,)
        
        # Compute k1 (slope at the beginning of the interval)
        k1 = output * dt
        
        # Estimate the mid-point value using k1
        mid_input = last_input + 0.5 * k1
        
        # Compute the model output at the mid-point
        mid_input_expanded = torch.cat([inputs, mid_input.unsqueeze(1)], dim=1)
        mid_output = model(mid_input_expanded)[:, -1]  # shape: (batch_size,)
        
        # Compute k2 (slope at the end of the interval)
        k2 = mid_output * dt
        
        # Compute the new output using the 2nd order Runge-Kutta formula
        new_output = last_input + k2
        
        # Add the new output to the inputs tensor
        inputs = torch.cat([inputs, new_output.unsqueeze(1)], dim=1)
    
    return inputs

def generate_idfm_2_ada(model, inputs, time_points=16, tol=1e-5, max_dt=0.1, min_dt=1e-4):
    """Generate using adaptive 2nd order Runge-Kutta (Midpoint) method"""
    dt = max_dt  # Start with a maximum initial step size
    current_time_points = 1 / (time_points - 1)
    while inputs.shape[1] < time_points:
        # Get the last input and its corresponding model output
        last_input = inputs[:, -1]  # shape: (batch_size,)
        output = model(inputs)[:, -1]  # shape: (batch_size,)
        
        # Compute k1 (slope at the beginning of the interval)
        k1 = output * dt
        
        # Estimate the mid-point value using k1
        mid_input = last_input + 0.5 * k1
        
        # Compute the model output at the mid-point
        mid_input_expanded = torch.cat([inputs, mid_input.unsqueeze(1)], dim=1)
        mid_output = model(mid_input_expanded)[:, -1]  # shape: (batch_size,)
        
        # Compute k2 (slope at the end of the interval)
        k2 = mid_output * dt
        
        # Compute the new output using the 2nd order Runge-Kutta formula
        new_output = last_input + k2
        
        # Estimate the error (difference between two successive methods)
        half_step_input = last_input + 0.5 * k1
        half_step_input_expanded = torch.cat([inputs, half_step_input.unsqueeze(1)], dim=1)
        half_step_output = model(half_step_input_expanded)[:, -1]  # shape: (batch_size,)
        
        k1_half = half_step_output * (0.5 * dt)
        half_step_new_output = half_step_input + k1_half
        
        # Estimate error
        error = torch.abs(new_output - half_step_new_output).max()

        # Adaptive step size control
        if error < tol:
            # If error is within tolerance, accept the step
            inputs = torch.cat([inputs, new_output.unsqueeze(1)], dim=1)
            current_time_points += dt
            if current_time_points >= time_points - 1:
                break
            # Increase the step size for the next iteration
            dt = min(max_dt, dt * 1.5)
        else:
            # Reduce the step size
            dt = max(min_dt, dt * 0.5)

    return inputs


def save_model(model, step, run_name, directory="/home/dfl32/scratch/training-runs/idfm/"):
    # Ensure the directory exists
    dir_path = os.path.join(directory, run_name)
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, f"checkpoint-{step}.pt")
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")

def plot_inf_traj(inf_traj):
    plt.figure(figsize=(6,6))
    plt.axis("equal")
    for i in range(inf_traj.shape[0]):
        #Plot the trajectories
        plt.plot(inf_traj.cpu().detach()[i, :, 0], inf_traj.cpu().detach()[i, :, 1], marker='', linestyle='-', linewidth=0.1, color='gray')
        # Mark starting point in red
        plt.plot(inf_traj.cpu().detach()[i, 0, 0], inf_traj.cpu().detach()[i, 0, 1], marker='o', markersize=1, color='red', label='Start' if i == 0 else "")
        # Mark ending point in blue
        plt.plot(inf_traj.cpu().detach()[i, -1, 0], inf_traj.cpu().detach()[i, -1, 1], marker='o', markersize=1, color='blue', label='End' if i == 0 else "")
    plt.title("Inference trajectories")
    return plt

def plot_attentions(attentions):
    average_attentions = [att.mean(dim=(0, 1)).cpu().numpy() for att in attentions]

    wandb_images = []
    # Plotting heatmaps for each layer
    for i, avg_att in enumerate(average_attentions):
        plt.figure(figsize=(8, 8))
        sns.heatmap(avg_att, cmap='viridis', annot=False)
        plt.title(f'Average Attention Matrix - Layer {i+1}')
        plt.xlabel('Sequence Position')
        plt.ylabel('Sequence Position')
        
        wandb_images.append(wandb.Image(plt, caption=f"Attention Layer {i}"))
        plt.close()
    return wandb_images

def plot_final_hidden_states(last_hidden_states):
    # Reshape the tensor to shape [256*15, 64]
    flattened_tensor = last_hidden_states.view(-1, last_hidden_states.shape[-1]).cpu().numpy()

    # Create labels for each vector based on their middle dimension (which repeats 256 times)
    labels = np.repeat(np.arange(last_hidden_states.shape[1]), last_hidden_states.shape[0])

    # Apply UMAP to reduce to 2D
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = umap_model.fit_transform(flattened_tensor)

    # Plot the UMAP result, colored by the middle dimension
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=1)
    plt.colorbar(scatter, label='Time')
    plt.title('UMAP of 64-Dimensional Vectors')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    return plt

def concat_time(input, batch_size, time_points, device):
    random_values = torch.rand(batch_size, time_points-2, 1)  # Generate 14 random values (since the first and last are fixed)

    # Sort the random values along the second dimension
    random_sorted = torch.sort(random_values, dim=1).values

    # Prepend 0.0 and append 1.0 to create the full sequence
    random_increasing_sequence = torch.cat((torch.zeros(batch_size, 1, 1), random_sorted, torch.ones(batch_size, 1, 1)), dim=1).to(device)

    # Concatenate along the last dimension to get the shape (4096, 16, 3)
    concat_tensor = torch.cat((input, random_increasing_sequence), dim=-1).to(device)
    return concat_tensor

def orthogonality_regularizer(hidden_states):
    """
    Computes the orthogonality regularization loss to ensure that hidden states at different positions 
    in the sequence have distinct representations.
    
    Args:
        hidden_states (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_size) containing
                                      the hidden states of a sequence.
                                      
    Returns:
        torch.Tensor: The regularization loss value.
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    
    # Normalize the hidden states for cosine similarity
    hidden_states_normalized = normalize(hidden_states, p=2, dim=-1)
    
    # Compute the pairwise dot products
    pairwise_dot_products = torch.bmm(hidden_states_normalized, hidden_states_normalized.transpose(1, 2))
    
    # Create the identity matrix as the target for MSE
    identity_matrix = torch.eye(seq_len, device=hidden_states.device).unsqueeze(0)  # Shape: (1, seq_len, seq_len)
    
    # Compute MSE between the pairwise dot products and the identity matrix
    loss = mse_loss(pairwise_dot_products, identity_matrix.expand(batch_size, -1, -1))
    
    return loss


def main(args):
    now = datetime.now()
    now = datetime.strftime(now, "%Y-%m-%d_%H-%M-%S")
    run_name = f"idfm-2moons-{args.timepoints}-{now}"
    device = torch.device("cuda")

    eight_gaussian_dataset = IFMdatasets(batch_size = args.batch_size, dataset_name="8gaussians", dim=2, gaussian_var=0.1)
    two_moon_dataset = IFMdatasets(batch_size = args.batch_size, dataset_name="2moons", dim=2)

    config = GPTNeoXConfig(
            hidden_size=args.hdim,
            intermediate_size=args.hdim*4,
            num_attention_heads=args.nhead,
            num_hidden_layers=args.nlayer,
            vocab_size=100,
            use_flash_attention_2=True,
            attention_dropout=args.attn_dropout
            )
    transformer = GPTNeoXForCausalLM(config).to(device)

    input_dim = 2 + args.continuous_time
    encoder = TwoLayerMLP(
        input_dim=input_dim,
        output_dim=args.hdim
    ).to(device)
    decoder = TwoLayerDecoder(args.hdim, 2).to(device)
    # decoder = nn.Linear(args.hdim, 2).to(device)

    model = IDFMModel(
        encoder,
        transformer,
        decoder
    )

    wandb.init(
            project="IFM",
            name=run_name,
        )
    wandb.watch(model, log="all", log_freq=10)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for step in tqdm(range(args.num_steps)):
        # ---- Sample data
        optimizer.zero_grad()
        x0 = eight_gaussian_dataset.generate_data(batch_size=args.batch_size).to(device)
        x1 = two_moon_dataset.generate_data(batch_size=args.batch_size).to(device)

        if args.ve:
            model_inputs, labels = generate_VE_paths(
                x0,
                x1,
                time_points=args.timepoints,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                use_last=args.use_last,
                linear_sched=args.linear_sched
            )
        else: 
            model_inputs, labels = generate_straight_paths(
                x0,
                x1,
                time_points=args.timepoints,
                path_sigma=args.path_sigma
            )

        if args.continuous_time:
            model_inputs = concat_time(model_inputs, args.batch_size, args.timepoints, device)
        outputs, _, last_hidden_states = model(model_inputs, output_attentions=args.orth_reg)
        pred = outputs[:, :-1, :]
        gt = labels[:,1:,:]

        loss = mse_loss(pred, gt, reduction="none").sum(dim=-1).mean()

        if args.orth_reg:
            orth_reg = orthogonality_regularizer(last_hidden_states)
            loss += args.or_weight*orth_reg
        loss.backward()
        optimizer.step()

        # Log to wandb
        if (step+1) % args.wandb_log_steps == 0:
            samples = eight_gaussian_dataset.generate_data(batch_size=args.num_inf_trajs).unsqueeze(1).to("cuda")
            model.eval()
            wandb_dict = {"loss": loss.item()}
            with torch.no_grad():
                num_inf_steps = max(args.num_inf_trajs//args.inf_batch_size, 1)
                all_trajs = []
                for step in range(num_inf_steps):
                    x0 = samples[step*args.inf_batch_size:(step+1)*args.inf_batch_size]
                    if args.use_rk:
                        if args.ada:
                            inf_traj = generate_idfm_2_ada(model, x0, time_points=args.timepoints, tol=args.tol, min_dt=args.min_dt)
                        else:
                            inf_traj = generate_idfm_2(model, x0, time_points=args.timepoints)
                    else:
                        inf_traj, attentions, last_hidden_states = generate_idfm(
                            model,
                            x0,
                            args.inf_batch_size,
                            time_points=args.timepoints,
                            continuous_time=args.continuous_time,
                            output_attentions=args.output_attentions
                            )
                        if args.output_attentions:
                            heatmaps = plot_attentions(attentions)
                            for i in range(len(heatmaps)):
                                wandb_dict[f"plots/Attention Layer {i}"] = heatmaps[i]
                            umap_plot = plot_final_hidden_states(last_hidden_states)
                            wandb_dict["plots/Last hidden states"] = wandb.Image(umap_plot)
                    all_trajs.append(inf_traj)
                all_trajs = torch.cat(all_trajs, dim=0)

                plt = plot_inf_traj(all_trajs)
                wandb_dict["plots/plot"] = wandb.Image(plt)

            wandb.log(wandb_dict)
            logger.info(f"Loss = {loss.item()}")

            plt.close()

            model.train()


        if (step+1) % args.save_steps == 0:
            save_model(model, step+1, run_name)

if __name__ == "__main__":
    args = parse_arguments()
    logger.info(args)
    main(args)