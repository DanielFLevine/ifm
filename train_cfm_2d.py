import argparse
import logging
import os
from datetime import datetime
from tqdm import tqdm

import torch
import wandb
import matplotlib.pyplot as plt
from torch import nn
from torch.nn.functional import mse_loss
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from utils.toy_datasets import IFMdatasets

from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *

logging.basicConfig(format='[%(levelname)s:%(asctime)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class IDFMModel(nn.Module):

    def __init__(self, encoder, transformer, decoder):
        super().__init__()
        self.encoder = encoder
        self.transformer = transformer
        self.decoder = decoder

    def forward(self, inputs):
        embs = self.encoder(inputs)
        hid_states = self.transformer.gpt_neox(inputs_embeds=embs).last_hidden_state
        outputs = self.decoder(hid_states)
        return outputs

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
        "--path_sigma",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--ode_solver",
        type=str,
        default='euler'
    )
    parser.add_argument(
        "--type",
        type=str,
        default='default'
    )
    return parser.parse_args()

def save_model(model, step, run_name, directory="/home/dfl32/scratch/training-runs/cfm/"):
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

def generate_idfm(model, inputs, batch_size, time_points=16, device='cuda'):
    """Generate using Eulers method
    """
    dt = 1/(time_points-1)
    inputs = inputs.unsqueeze(1)
    t = torch.linspace(0.0, 1.0, time_points).view(1, time_points, 1).expand(batch_size, time_points, 1).to(device)
    for i in range(time_points-1):
        model_inputs = torch.cat([inputs[:, i], t[:, i]], dim=-1)
        outputs = model(model_inputs)
        new_output = inputs[:, -1] + outputs*dt
        inputs = torch.cat([inputs, new_output.unsqueeze(1)], dim=1)
    return inputs


def main(args):
    logger.info(f"CFM Method {args.type}")
    assert args.type in ('default', 'ot', 'sb'), f"'--type' flag needs to be in ('default', 'ot', 'sb'). Currently {args.type}"
    now = datetime.now()
    now = datetime.strftime(now, "%Y-%m-%d_%H-%M-%S")
    run_name = f"cfm-2moons-tp{args.timepoints}-{args.type}-{now}"
    device = torch.device("cuda")

    eight_gaussian_dataset = IFMdatasets(batch_size = args.batch_size, dataset_name="8gaussians", dim=2, gaussian_var=0.1)
    two_moon_dataset = IFMdatasets(batch_size = args.batch_size, dataset_name="2moons", dim=2)

    model = MLP(dim=2, time_varying=True).to(device)
    
    wandb.init(
            project="IFM",
            name=run_name,
        )
    wandb.watch(model, log="all", log_freq=10)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.type == 'ot':
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=args.path_sigma)
    elif args.type == 'sb':
        FM = SchrodingerBridgeConditionalFlowMatcher(sigma=args.path_sigma, ot_method="exact")
    else:
        FM = ConditionalFlowMatcher(sigma=args.path_sigma)

    for step in tqdm(range(args.num_steps)):
        optimizer.zero_grad()
        x0 = eight_gaussian_dataset.generate_data(batch_size=args.batch_size).to(device)
        x1 = two_moon_dataset.generate_data(batch_size=args.batch_size).to(device)
        
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        # else:
        #     t = torch.rand(x0.shape[0]).type_as(x0).to(device)
        #     xt = sample_conditional_pt(x0, x1, t, sigma=args.path_sigma).to(device)
        #     ut = compute_conditional_vector_field(x0, x1).to(device)

        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        loss = torch.mean((vt - ut) ** 2)

        loss.backward()
        optimizer.step()

        # Log to wandb
        if (step+1) % args.wandb_log_steps == 0:
            # node = NeuralODE(torch_wrapper(model), solver=args.ode_solver, sensitivity="adjoint", atol=1e-4, rtol=1e-4)
            with torch.no_grad():
                # inf_traj = node.trajectory(
                #     sample_8gaussians(args.inf_batch_size).to(device),
                #     t_span=torch.linspace(0, 1, args.timepoints).to(device),
                # )
                # plt = plot_inf_traj(inf_traj.transpose(0, 1))
                inputs = eight_gaussian_dataset.generate_data(batch_size=args.inf_batch_size).to(device)
                inf_traj = generate_idfm(model, inputs, args.inf_batch_size, time_points=args.timepoints)
                plt = plot_inf_traj(inf_traj)
            wandb.log({
                "plot": wandb.Image(plt),
                "loss": loss.item(),
                })
            plt.close()
            logger.info(f"Loss = {loss.item()}")


        if (step+1) % args.save_steps == 0:
            save_model(model, step+1, run_name)

if __name__ == "__main__":
    args = parse_arguments()
    logger.info(args)
    main(args)