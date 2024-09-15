import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from tqdm import tqdm
import wandb
from itertools import cycle

logging.basicConfig(format='[%(levelname)s:%(asctime)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim * 2)

    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))
        return self.fc2(x)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class CustomDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return torch.tensor(self.dataset[idx]['expr'], dtype=torch.float32)

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE, KLD

def save_model(model, step, run_name, directory="/home/dfl32/scratch/training-runs/vae/"):
    dir_path = os.path.join(directory, run_name)
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, f"checkpoint-{step}.pt")
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")

def run_eval_loop(model, val_dataloader, device):
    model.eval()
    mse_losses = []
    kl_losses = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            batch = batch.to(device)
            recon_batch, mu, logvar = model(batch)
            mse_loss, kl_loss = loss_function(recon_batch, batch, mu, logvar)
            mse_losses.append(mse_loss.item())
            kl_losses.append(kl_loss.item())
    eval_mse_loss = sum(mse_losses) / len(mse_losses)
    eval_kl_loss = sum(kl_losses) / len(kl_losses)
    eval_total_loss = eval_mse_loss + eval_kl_loss
    model.train()
    return eval_mse_loss, eval_kl_loss, eval_total_loss

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_dataset_path", type=str, default="/path/to/dataset")
    parser.add_argument("--output_dir", type=str, default="/path/to/output")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_train_steps", type=int, default=100000)
    parser.add_argument("--input_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--wandb_log_steps", type=int, default=100)
    parser.add_argument("--eval_dataset_size", type=int, default=1000)
    parser.add_argument("--beta", type=float, default=1.0)
    return parser.parse_args()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = VAE(args.input_dim, args.hidden_dim, args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    dataset = load_from_disk(args.llm_dataset_path)
    train_dataset = CustomDataset(dataset)
    val_dataset = CustomDataset(dataset.select(range(min(len(dataset), args.eval_dataset_size))))

    train_dataloader = cycle(DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True))
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    run_name = f"vae-{args.input_dim}-{args.hidden_dim}-{args.latent_dim}"
    wandb.init(project="VAE_Project", name=run_name)
    wandb.watch(model, log="all", log_freq=10)

    for step in tqdm(range(args.num_train_steps)):
        batch = next(train_dataloader)
        batch = batch.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        mse_loss, kl_loss = loss_function(recon_batch, batch, mu, logvar)
        loss = mse_loss + args.beta * kl_loss
        loss.backward()
        optimizer.step()

        if (step + 1) % args.wandb_log_steps == 0:
            eval_mse_loss, eval_kl_loss, eval_total_loss = run_eval_loop(model, val_dataloader, device)
            wandb.log({
                "loss": loss.item(),
                "mse_loss": mse_loss.item(),
                "kl_loss": kl_loss.item(),
                "eval_mse_loss": eval_mse_loss,
                "eval_kl_loss": eval_kl_loss,
                "eval_total_loss": eval_total_loss
            })
            logger.info(f"Step {step + 1}, Loss: {loss.item()}, MSE Loss: {mse_loss.item()}, KL Loss: {kl_loss.item()}, Eval MSE Loss: {eval_mse_loss}, Eval KL Loss: {eval_kl_loss}, Eval Total Loss: {eval_total_loss}")

        if (step + 1) % args.wandb_log_steps == 0:
            save_model(model, step + 1, run_name)

if __name__ == "__main__":
    args = parse_arguments()
    logger.info(args)
    main(args)