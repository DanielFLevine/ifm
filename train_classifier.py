import os
import pickle
import argparse
from itertools import cycle

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import scanpy as sc
from sklearn.model_selection import train_test_split
import wandb
import scipy.sparse
from utils.metrics import transform_gpu  # Assuming transform_gpu is defined in metrics.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple MLP on PCA-transformed data")
    parser.add_argument('--num_steps', type=int, default=1000, help='Number of steps (minibatches) to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension of the MLP')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--leiden', type=int, default=2, help='Learning rate for the optimizer')
    parser.add_argument('--steps_per_checkpoint', type=int, default=1000, help='Number of steps per checkpoint')
    parser.add_argument('--val_loss_interval', type=int, default=100, help='Number of steps between validation loss computations')
    return parser.parse_args()

# Define the model
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def main():
    args = parse_args()

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    leiden = f"0{args.leiden}" if args.leiden < 10 else f"{args.leiden}"
    # Load the data
    adata = sc.read_h5ad(f'/home/dfl32/project/ifm/cinemaot_data/conditional_cinemaot/integrated_ifm_leiden_{leiden}.h5ad')
    labels = adata.obs['leiden'].astype('category').cat.codes.values

    # # Preprocess the data
    # sc.pp.filter_cells(adata, min_genes=200)
    # sc.pp.filter_genes(adata, min_cells=3)
    # adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    # adata = adata[adata.obs.pct_counts_mt < 5, :].copy()
    # sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)

    # # Combine the three columns into a single label
    # adata.obs['combined_label'] = adata.obs[['cell_type0528', 'perturbation', 'chronicity']].astype(str).agg('_'.join, axis=1)
    # labels = adata.obs['combined_label'].astype('category').cat.codes.values

    # Load the PCA model
    with open('/home/dfl32/project/ifm/projections/pcadim1000_numsamples10000.pickle', 'rb') as f:
        pca_model = pickle.load(f)

    # Apply PCA transformation on GPU
    expression_data = adata.X.toarray() if isinstance(adata.X, scipy.sparse.spmatrix) else adata.X
    pca_data = transform_gpu(expression_data, pca_model)

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(pca_data, labels, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Create DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = 1000
    output_dim = len(set(labels))

    model = SimpleMLP(input_dim, args.hidden_dim, output_dim).to(device)

    # Calculate class weights
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Define the loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Initialize wandb
    wandb.init(project="MLP_Classification", config=args)
    wandb.watch(model, log="all")

    # Training loop
    num_steps = args.num_steps
    steps_per_checkpoint = args.steps_per_checkpoint
    val_loss_interval = args.val_loss_interval

    best_val_loss = float('inf')
    checkpoint_dir = f'/home/dfl32/scratch/unconditional_classifier_combined_labels/checkpoints_hidden_dim_{args.hidden_dim}_{args.leiden}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_iter = cycle(train_loader)
    for step in tqdm(range(num_steps)):
        model.train()
        inputs, targets = next(train_iter)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()

        # Save checkpoint
        if (step + 1) % steps_per_checkpoint == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step{step+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Saved checkpoint: {checkpoint_path}')

        if (step + 1) % val_loss_interval == 0:
            model.eval()
            val_loss = 0.0
            class_losses = {label: 0.0 for label in range(output_dim)}
            class_counts = {label: 0 for label in range(output_dim)}
            class_correct = {label: 0 for label in range(output_dim)}
            with torch.no_grad():
                for inputs, targets in tqdm(val_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    
                    # Calculate per-class loss and accuracy
                    _, predicted = torch.max(outputs, 1)
                    for i, target in enumerate(targets):
                        class_losses[target.item()] += loss.item()
                        class_counts[target.item()] += 1
                        if predicted[i] == target:
                            class_correct[target.item()] += 1

            val_loss /= len(val_loader.dataset)
            for label in class_losses:
                if class_counts[label] > 0:
                    class_losses[label] /= class_counts[label]

            class_accuracies = {label: class_correct[label] / class_counts[label] if class_counts[label] > 0 else 0 for label in class_correct}

            print(f'Step {step+1}/{num_steps}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Plot per-class loss
            plt.figure(figsize=(10, 6))
            sns.barplot(x=list(class_losses.keys()), y=list(class_losses.values()))
            plt.xlabel('Class')
            plt.ylabel('Loss')
            plt.title('Per-Class Validation Loss')
            per_class_loss_chart = wandb.Image(plt)
            plt.close()

            # Plot per-class accuracy
            plt.figure(figsize=(10, 6))
            sns.barplot(x=list(class_accuracies.keys()), y=list(class_accuracies.values()))
            plt.xlabel('Class')
            plt.ylabel('Accuracy')
            plt.title('Per-Class Validation Accuracy')
            per_class_accuracy_chart = wandb.Image(plt)
            plt.close()

            # Log everything together
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "class_losses": class_losses,
                "class_accuracies": class_accuracies,
                "per_class_loss_chart": per_class_loss_chart,
                "per_class_accuracy_chart": per_class_accuracy_chart
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')

    # Load the best model and evaluate on the test set
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / total

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}')
    wandb.log({"test_loss": test_loss, "test_accuracy": accuracy})

    def plot_confusion_matrix(y_true, y_pred, labels, save_path):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.savefig(save_path)
        plt.close()

    def sample_and_evaluate(model, data_loader, device, num_samples_per_class=100):
        model.eval()
        all_preds = []
        all_targets = []
        class_counts = {label: 0 for label in range(len(set(labels)))}
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                for pred, target in zip(predicted, targets):
                    if class_counts[target.item()] < num_samples_per_class:
                        all_preds.append(pred.item())
                        all_targets.append(target.item())
                        class_counts[target.item()] += 1
                    if all(count >= num_samples_per_class for count in class_counts.values()):
                        break
                if all(count >= num_samples_per_class for count in class_counts.values()):
                    break
        return all_targets, all_preds

    # Sample 100 cells from each label in the test set and compute accuracy
    test_targets, test_preds = sample_and_evaluate(model, test_loader, device, num_samples_per_class=100)

    # Plot and save the confusion matrix
    labels_list = list(range(len(set(labels))))
    plot_confusion_matrix(test_targets, test_preds, labels_list, "/home/dfl32/project/ifm/plots/confusion_matrix.png")

if __name__ == "__main__":
    main()