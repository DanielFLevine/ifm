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
from torch.nn import MSELoss
from torch.nn.functional import mse_loss, normalize
from torch.distributions import Normal, kl_divergence
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from utils.toy_datasets import IFMdatasets
from utils.modules import CustomVAEDecoder, TwoLayerMLP, TwoLayerDecoder

logging.basicConfig(format='[%(levelname)s:%(asctime)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class CaLMFlowModel(nn.Module):

    def __init__(self, encoder, transformer, decoder, use_vae=True):
        super().__init__()
        self.encoder = encoder
        self.transformer = transformer
        self.decoder = decoder
        self.use_vae = use_vae

    def forward(self, inputs, output_attentions=False, temp=1.0, pps=1):
        embs = self.encoder(inputs)
        hid_states = self.transformer.gpt_neox(inputs_embeds=embs, output_attentions=output_attentions)
        if self.use_vae:
            outputs, latents, cond_dist = self.decoder(hid_states.last_hidden_state, temperature=temp)
            return outputs, latents, cond_dist
        else:
            outputs = self.decoder(hid_states.last_hidden_state)
        if output_attentions:
            return outputs, hid_states.attentions, hid_states.last_hidden_state
        return outputs, None, None

    def multipoint_reshape(self, X, points_per_sample):

        batch_size, seq_len, feature_dim = X.shape

        new_batch_size = batch_size//points_per_sample

        # Reshape and permute the tensor
        x_reshaped = X.view(new_batch_size, points_per_sample, seq_len, feature_dim).permute(0, 2, 1, 3).reshape(new_batch_size, points_per_sample*seq_len, feature_dim)
        return x_reshaped

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
    parser.add_argument(
        "--use_vae",
        action="store_true",
        help="Use VAE decoder"
    )
    parser.add_argument(
        "--mlp_musig",
        action="store_true",
        help="Use mlps to project hidden state to mean and variance"
    )
    parser.add_argument(
        "--pps",
        type=int,
        default=1,
        help="Number of trajectories per sample (not compatible with idfm; set to 1 for idfm)"
    )
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=1.0,
        help="Weight for KL divergence regularizer in loss function"
    )
    parser.add_argument(
        "--space_dim",
        type=int,
        default=1,
        help="Number of tokens per time point"
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=1.0,
        help="Temperature for inference with VAE"
    )
    parser.add_argument(
        "--generate_inputs",
        action="store_true",
        help="Use generated trajectories as input during training"
    )
    parser.add_argument(
        "--random_labels",
        action="store_true",
        help="Uses random labels for intermediate timepoints in idfm"
    )
    parser.add_argument(
        "--prob_weight",
        type=float,
        default=1.0,
        help="Weighting for log probabilities term in loss"
    )
    parser.add_argument(
        "--ve_inputs",
        action="store_true",
        help="Uses ve paths for inputs when using random labels for idfm"
    )
    parser.add_argument(
        "--plot_conditional_inf",
        action="store_true",
        help="Plots using the conditional inference method"
    )
    parser.add_argument(
        "--square_probs",
        action="store_true",
        help="Squares probabilities used if using random labels"
    )
    parser.add_argument(
        "--add_noise_to_gen",
        action="store_true",
        help="Adds noise to generated inputs"
    )
    parser.add_argument(
        "--gen_input_noise",
        type=float,
        default=0.1,
        help="Amount of noise to generated inputs"
    )
    parser.add_argument(
        "--add_noise_to_input",
        action="store_true",
        help="Adds noise to inputs while generating outputs"
    )
    parser.add_argument(
        "--plot_train_trajs",
        action="store_true",
        help="Plot sample training trajectories"
    )
    parser.add_argument(
        "--power",
        action="store_true",
        help="Raise training trajectories to randomly sampled powers to create curved training trajectories"
    )
    parser.add_argument(
        "--predict_point",
        action="store_true",
        help="Predicts x1 instead of x1-x0"
    )
    parser.add_argument(
        "--scale_denom",
        action="store_true",
        help="Scales the target point by 1/(1-t) when using 'predict_point'"
    )
    parser.add_argument(
        "--predict_solver_step",
        action="store_true",
        help="Model directly predicts Euler solver step"
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=1.0,
        help="Amount to scale inference step size by"
    )
    return parser.parse_args()

def interpolate_tensors(A, B, time_points):
    # Ensure A and B have the same shape
    assert A.shape == B.shape, "A and B must have the same shape"
    
    # Get batch size and feature dimensions
    batch_size, feature_dim = A.shape
    
    # Initialize the new tensor
    new_tensor = torch.zeros((batch_size, time_points, feature_dim), device=A.device)
    
    # Set the first and last time points
    new_tensor[:, 0, :] = A
    new_tensor[:, time_points-1, :] = B
    
    # Interpolate the intermediate time points
    for i in range(1, time_points-1):
        t_i = i / (time_points - 1)  # Calculate the interpolation factor
        new_tensor[:, i, :] = A * (1 - t_i) + B * t_i
    
    return new_tensor


def multipoint_reshape(X, points_per_sample):

    batch_size, seq_len, feature_dim = X.shape

    new_batch_size = batch_size//points_per_sample

    # Reshape and permute the tensor
    x_reshaped = X.view(new_batch_size, points_per_sample, seq_len, feature_dim).permute(0, 2, 1, 3).reshape(new_batch_size, points_per_sample*seq_len, feature_dim)
    return x_reshaped

def unipoint_reshape(X, points_per_sample):

    batch_size, seq_len, feature_dim = X.shape

    new_batch_size = batch_size*points_per_sample

    # Reshape and permute the tensor
    x_reshaped = X.reshape(batch_size, seq_len//points_per_sample, points_per_sample, feature_dim).permute(0, 2, 1, 3).reshape(new_batch_size, seq_len//points_per_sample, feature_dim)

    return x_reshaped

def generate_random_labels(X, dataset_class, device='cuda', time_points=16):
    all_labels = []
    for i in range(time_points):
        labels = dataset_class.generate_data(batch_size=X.shape[0]).to(device) # shape batch_size x feature_dim
        all_labels.append(labels)
    
    all_labels = torch.stack(all_labels, dim=1).to(device)
    
    return all_labels
    
def generate_straight_paths(
    X,
    Y,
    device='cuda',
    time_points=16,
    path_sigma=0.0,
    idfm=False,
    random_labels=False,
    power=False,
    predict_point=False
    ):
    W = torch.zeros(X.shape[0], time_points, X.shape[1]).to(device)
    eps = 1e-5
    if random_labels:
        means = torch.zeros(X.shape[0], time_points, X.shape[1]).to(device)
        sigmas = torch.full((X.shape[0], time_points, X.shape[1]), path_sigma+0.1).to(device)

    # Define the time points, evenly spaced from 0 to 1, inclusive
    times = torch.linspace(0, 1, time_points).to(device)

    # Generate W where each slice W[:, i, :] is based on the linear combination of X and Y
    if power:
        batch_size = X.shape[0]
        power_tensor = eps + torch.rand(batch_size).to(device) * (5 - eps)
        power_tensor = power_tensor.view(batch_size, 1)
    for i, t in enumerate(times):
        if (path_sigma > 0.0):

            variance = (((1-t)+eps)*path_sigma)**2
            # Calculate the convex combination of X and Y
            mean = (1 - t) * X + t * Y

            # Sample from a Gaussian with the calculated mean and variance
            if power:
                t = torch.tensor(t).repeat(batch_size, 1).to(device)
                W[:, i, :] = (((1 - t)**power_tensor) * X + (t**power_tensor) * Y) + torch.randn_like(X).to(device) * variance
            else:
                W[:, i, :] = mean + torch.randn_like(X).to(device) * variance
            if random_labels:
                means[:, i, :] = mean
                sigmas[:, i, :] = torch.full_like(mean, path_sigma).to(device)
        elif power:
            if random_labels:
                mean = (1 - t) * X + t * Y
                means[:, i, :] = mean
            batch_size = X.shape[0]

            # Reshape t and power_tensor to allow broadcasting (if necessary)
            t = torch.tensor(t).repeat(batch_size, 1).to(device)

            # Calculate the modified convex combination
            W[:, i, :] = ((1 - t)**power_tensor) * X + (t**power_tensor) * Y
        else:
            W[:, i, :] = (((1 - t) * X) + (t * Y)).to(device)
    
    W = W.to(dtype=torch.float32)
    if idfm:
        dist = None
        if predict_point:
            labels = Y.unsqueeze(1).repeat(1,time_points,1)
        else:
            labels = (Y.unsqueeze(1) - X.unsqueeze(1)).repeat(1,time_points,1)
        if random_labels:
            dist = Normal(loc=means, scale=sigmas)
        return W, labels, dist
    
    labels = W.detach().clone()
    return W, labels, None

def generate_VE_paths(X, Y, device='cuda', time_points=16, sigma_min=1e-4, sigma_max=1e1, use_last=False, linear_sched=False, idfm=False):
    sigma_min = torch.tensor(sigma_min)
    sigma_max = torch.tensor(sigma_max)

    W = torch.zeros(X.shape[0], time_points, X.shape[1]).to(device)

    # Define the time points, evenly spaced from 0 to 1, inclusive
    times = torch.linspace(0, 1, time_points).to(device)

    if idfm:
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

        # log_term = torch.log(sigma_max / sigma_min)
        # derivative = -sigma_max * log_term * (sigma_max / sigma_min) ** (-t)
        
        # # Divide the derivative by the original variance
        # ve_scale = derivative / variance
        if idfm:
            labels[:, i, :] = labels[:, i, :]
    
    if not idfm:
        labels = W.detach().clone()
    return W, labels

def generate_ifm(model, inputs, time_points=16, space_dim=1, temp=1.0, pps=1, continuous_time=False, device='cuda', output_attentions=False, use_vae=True):
    for _ in range(time_points-1):
        outputs = model.encoder(inputs)

        # Reshape for spatial integration
        batch_size, seq_len, feature = outputs.shape
        outputs = outputs.view(batch_size, seq_len, space_dim, feature // space_dim)
        outputs = outputs.view(batch_size, seq_len * space_dim, feature // space_dim)


        outputs = model.transformer.gpt_neox(inputs_embeds=outputs).last_hidden_state

        if not use_vae:
            outputs = model.decoder(outputs)
        else:
            outputs, _, _ = model.decoder(outputs, temperature=temp)
        last_outputs = outputs[:, -pps:, :]
        inputs = torch.concat([inputs, last_outputs], axis=1)
    return inputs

def generate_idfm(
    model,
    inputs,
    batch_size,
    time_points=16,
    continuous_time=False,
    device='cuda',
    output_attentions=False,
    add_noise_to_gen=False,
    add_noise_to_input=False,
    predict_point=False,
    scale_denom=False,
    predict_solver_step=False,
    noise=0.1,
    input_noise=0.0,
    scale_factor=1.0
    ):
    """Generate using Eulers method
    """
    dt = 1/(time_points-1)*scale_factor
    if continuous_time:
        t = torch.linspace(0.0, 1.0, time_points).view(1, time_points, 1).expand(batch_size, time_points, 1).to(device)
    for i in range(time_points-1):
        if continuous_time:
            inputs = torch.cat([inputs, t[:, :i+1]], dim=-1)
            outputs, attentions, last_hidden_states = model(inputs, output_attentions=output_attentions)
            new_output = inputs[:, -1, :-1] + outputs[:, -1]*dt
            inputs = torch.cat([inputs[:, :, :-1], new_output.unsqueeze(1)], dim=1)
        else:
            model_inputs = inputs
            if add_noise_to_input:
                model_inputs = inputs + torch.randn_like(inputs).to(device) * input_noise
            outputs, attentions, last_hidden_states = model(model_inputs, output_attentions=output_attentions)
            if predict_point:
                # move towards x1 at time t by taking a step ((x1-xt)/(1-t))*dt
                dx = outputs[:, -1]-inputs[:, -1]
                time_scale = 1.0
                if scale_denom:
                    time_scale = 1.0/(1.0-((i+1)/time_points))
                new_output = inputs[:, -1] + dx*dt*time_scale
            elif predict_solver_step:
                new_output = outputs[:, -1]
            else:
                new_output = inputs[:, -1] + outputs[:, -1]*dt
            inputs = torch.cat([inputs[:, :], new_output.unsqueeze(1)], dim=1)
    if add_noise_to_gen:
        inputs = inputs + torch.randn_like(inputs).to(device) * noise
    if output_attentions:
        return inputs, attentions, last_hidden_states
    return inputs, None, None

def generate_cond_idfm(model, inputs, batch_size, time_points=16, device='cuda', output_attentions=False):
    """Generate using Eulers method
    """
    dt = 1/(time_points-1)
    model_inputs = inputs
    trajectories = inputs
    for i in range(time_points-1):
        outputs, attentions, last_hidden_states = model(model_inputs, output_attentions=output_attentions)
        new_output = model_inputs[:, -1] + outputs[:, -1]*dt
        trajectories = torch.cat([trajectories[:, :], new_output.unsqueeze(1)], dim=1)
        # logger.info(f"\nINPUT SHAPE {inputs.shape}")
        # logger.info(f"\nNEW OUTPUT SHAPE {new_output.shape}")
        model_inputs = interpolate_tensors(inputs[:,0,:], new_output, i+2)

    if output_attentions:
        return trajectories, attentions, last_hidden_states
    return trajectories, None, None

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
    if args.idfm:
        run_name = f"idfm-2moons-tp{args.timepoints}-pathsig{args.path_sigma}-{now}"
    else:
        run_name = f"idfm-2moons-tp{args.timepoints}-pps{args.pps}-temp{args.temp}-{now}"
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
    if args.use_vae:
        decoder = CustomVAEDecoder(
            hidden_size=config.hidden_size,
            input_dim=input_dim,
            device=device,
            reshape_postvae=True,
            space_dim=args.space_dim,
            num_blocks=1,
            mlp_enc=args.mlp_musig
        )
    else:
        decoder = TwoLayerDecoder(args.hdim, 2).to(device)
    # decoder = nn.Linear(args.hdim, 2).to(device)

    model = CaLMFlowModel(
        encoder,
        transformer,
        decoder,
        use_vae=args.use_vae
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
                linear_sched=args.linear_sched,
                idfm=args.idfm
            )
        else: 
            model_inputs, labels, dist = generate_straight_paths(
                x0,
                x1,
                time_points=args.timepoints,
                path_sigma=args.path_sigma,
                idfm=args.idfm,
                random_labels=args.random_labels,
                power=args.power,
                predict_point=args.predict_point
            )

        if args.continuous_time:
            model_inputs = concat_time(model_inputs, args.batch_size, args.timepoints, device)
        if not args.idfm:
            if args.generate_inputs:
                with torch.no_grad():
                    x0 = x0.unsqueeze(1)
                    x0 = multipoint_reshape(x0, args.pps)
                    model_inputs = generate_ifm(model, x0, pps=args.pps, time_points=args.timepoints, use_vae=args.use_vae)
            model_inputs = multipoint_reshape(model_inputs, args.pps)
            labels = multipoint_reshape(labels, args.pps)
            outputs, latents, cond_dist = model(model_inputs, output_attentions=args.orth_reg)
            pred = outputs[:, :-args.pps, :].contiguous()
            gt = labels[:,args.pps:,:]
            loss_fn = MSELoss(reduction='none')
            recon_loss = loss_fn(pred, gt)
            if args.use_vae:
                pz = Normal(torch.zeros_like(latents), torch.ones_like(latents))
                kl_divergence_z = kl_divergence(
                    cond_dist,
                    pz
                ).sum(dim=-1)[:,:-args.space_dim*args.pps]
                recon_loss = torch.mean(recon_loss.sum(-1))
                kl_loss = torch.mean(kl_divergence_z)
                loss = recon_loss + (kl_loss*args.kl_weight)
            else:
                loss = torch.mean(recon_loss.sum(-1))
        else:
            probs = 1.0
            if args.random_labels:
                permutation = torch.randperm(model_inputs.size(0))
                if args.ve_inputs:
                    x0 = eight_gaussian_dataset.generate_data(batch_size=args.batch_size).to(device)
                    x1 = two_moon_dataset.generate_data(batch_size=args.batch_size).to(device)
                    model_inputs, _ = generate_VE_paths(
                        x0,
                        x1,
                        time_points=args.timepoints,
                        sigma_min=args.sigma_min,
                        sigma_max=args.sigma_max,
                        use_last=args.use_last,
                        linear_sched=args.linear_sched,
                        idfm=args.idfm
                    )
                elif args.generate_inputs:
                    with torch.no_grad():
                        x0 = eight_gaussian_dataset.generate_data(batch_size=args.batch_size).to(device)
                        x0 = x0.unsqueeze(1)
                        # x0 = multipoint_reshape(x0, args.pps)
                        model_inputs, _, _ = generate_idfm(
                            model,
                            x0,
                            x0.shape[0],
                            time_points=args.timepoints,
                            continuous_time=args.continuous_time,
                            add_noise_to_gen=args.add_noise_to_gen,
                            noise=args.gen_input_noise,
                            scale_factor=args.scale_factor
                        )
                else:
                    model_inputs = model_inputs[permutation]
                probs = dist.log_prob(model_inputs).exp()
                if args.square_probs:
                    probs = torch.pow(probs, 2.0)
            elif args.generate_inputs:
                with torch.no_grad():
                    x0 = x0.unsqueeze(1)
                    # x0 = multipoint_reshape(x0, args.pps)
                    model_inputs, _, _ = generate_idfm(
                        model,
                        x0,
                        x0.shape[0],
                        time_points=args.timepoints,
                        continuous_time=args.continuous_time,
                        add_noise_to_gen=args.add_noise_to_gen,
                        noise=args.gen_input_noise,
                        scale_factor=args.scale_factor
                    )
            outputs, _, last_hidden_states = model(model_inputs, output_attentions=args.orth_reg)
            pred = outputs[:, :-args.pps, :].contiguous()
            gt = labels
            if args.predict_solver_step:
                dt = 1/(args.timepoints-1)
                if args.predict_point:
                    time_scale = 1.0
                    # if args.scale_denom:
                    #     time_scale = 1.0/(1.0-((i+1)/args.timepoints))
                    gt = model_inputs + ((labels-model_inputs)*dt*time_scale)
                else:
                    gt = model_inputs + (labels*dt)
            gt = gt[:,args.pps:,:]
            loss = (mse_loss(pred, gt, reduction="none")*probs[:, :-args.pps, :]).sum(dim=-1).mean()


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

            if args.plot_train_trajs:
                X = eight_gaussian_dataset.generate_data(batch_size=64).to(device)
                Y = two_moon_dataset.generate_data(batch_size=64).to(device)
                train_paths, _, _ = generate_straight_paths(X, Y, time_points=args.timepoints, path_sigma=args.path_sigma, power=args.power, predict_point=args.predict_point)
                train_plt = plot_inf_traj(train_paths)
                wandb_dict["plots/train_plot"] = wandb.Image(train_plt)

            with torch.no_grad():
                num_inf_steps = max(args.num_inf_trajs//args.inf_batch_size, 1)
                all_trajs = []
                if args.add_noise_to_input:
                    all_noise_trajs = []
                for step in range(num_inf_steps):
                    x0 = samples[step*args.inf_batch_size:(step+1)*args.inf_batch_size]
                    if args.idfm:
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
                                    output_attentions=args.output_attentions,
                                    scale_factor=args.scale_factor,
                                    predict_point=args.predict_point,
                                    scale_denom=args.scale_denom,
                                    predict_solver_step=args.predict_solver_step
                                    )
                            if args.add_noise_to_input:
                                noise_inf_traj, _, _ = generate_idfm(
                                    model,
                                    x0,
                                    args.inf_batch_size,
                                    time_points=args.timepoints,
                                    continuous_time=args.continuous_time,
                                    output_attentions=args.output_attentions,
                                    add_noise_to_input=args.add_noise_to_input,
                                    input_noise=args.path_sigma,
                                    scale_factor=args.scale_factor,
                                    predict_point=args.predict_point,
                                    scale_denom=args.scale_denom,
                                    predict_solver_step=args.predict_solver_step
                                    )
                            if args.output_attentions:
                                heatmaps = plot_attentions(attentions)
                                for i in range(len(heatmaps)):
                                    wandb_dict[f"plots/Attention Layer {i}"] = heatmaps[i]
                                umap_plot = plot_final_hidden_states(last_hidden_states)
                                wandb_dict["plots/Last hidden states"] = wandb.Image(umap_plot)
                    else:
                        x0 = multipoint_reshape(x0, args.pps)
                        inf_traj = generate_ifm(model, x0, temp=args.temp, pps=args.pps, time_points=args.timepoints, use_vae=args.use_vae)
                        inf_traj = unipoint_reshape(inf_traj, args.pps)
                    if args.add_noise_to_input:
                        all_noise_trajs.append(noise_inf_traj)
                    all_trajs.append(inf_traj)
                all_trajs = torch.cat(all_trajs, dim=0)
                if args.add_noise_to_input:
                    all_noise_trajs = torch.cat(all_noise_trajs, dim=0)
                    noise_plt = plot_inf_traj(all_noise_trajs)
                    wandb_dict["plots/plot_with_noise_inputs"] = wandb.Image(noise_plt)


                if args.plot_conditional_inf:
                    num_inf_steps = max(args.num_inf_trajs//args.inf_batch_size, 1)
                    all_cond_trajs = []
                    for step in range(num_inf_steps):
                        x0 = samples[step*args.inf_batch_size:(step+1)*args.inf_batch_size]
                        inf_traj, attentions, last_hidden_states = generate_cond_idfm(
                            model,
                            x0,
                            args.inf_batch_size,
                            time_points=args.timepoints,
                            output_attentions=args.output_attentions
                        )
                        all_cond_trajs.append(inf_traj)
                    all_cond_trajs = torch.cat(all_cond_trajs, dim=0)
                    cond_plt = plot_inf_traj(all_cond_trajs)
                    wandb_dict["plots/cond_inf_plot"] = wandb.Image(cond_plt)
                    


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