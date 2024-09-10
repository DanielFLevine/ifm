import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
from ipdb import set_trace
import json
import numpy as np
from datasets import load_from_disk, concatenate_datasets
import os
from tqdm import tqdm


wandb.login()

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        # No need for softmax here as CrossEntropyLoss already applies it
        return x

def train_mlp_classifier(train_dataset, val_dataset, label_name, input_dim, hidden_dim, output_dim, pert_split, num_epochs=100, learning_rate=0.001, weight_decay=1e-5,batch_size=32):
   
    label_int = label_name + "_int"
    model = MLPClassifier(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Create DataLoader for minibatch training
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in tqdm(range(num_epochs), desc="Training Epochs", unit="epoch", leave=True):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for batch in train_loader:
            x = batch["expression"]
            y = batch[label_int]
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += y.size(0)
            correct_train += (predicted_train == y).sum().item()
            
            if use_wandb:
                wandb.log({"train_loss": loss.item()})
        
        train_accuracy = 100 * correct_train / total_train
        if use_wandb:
            wandb.log({"train_accuracy": train_accuracy,
                       "train_loss/epoch": train_loss})
                

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in val_loader:
                x = batch["expression"]
                y = batch[label_int]
                outputs = model(x)
                loss = criterion(outputs, y)
                if use_wandb:
                    wandb.log({"val_loss": loss.item()})
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                
            val_accuracy = 100 * correct / total
            if use_wandb:
                wandb.log({"val_accuracy": val_accuracy,
                           "val_loss/epoch": val_loss})
        tqdm.write(f'Train loss: {train_loss:.4f}; Train Accuracy: {train_accuracy:.4f};Validation Loss: {val_loss:.4f}: Validation Accuracy: {val_accuracy:.4f}')
    if use_wandb:
        wandb.finish()
 
            

    # Save the model at the end
    save_dir = f"/gpfs/radev/scratch/dijk/sh2748/calmflow_singlecell/classifier_models/{pert_split}"
    model_dir = label_name
    model_dir = os.path.join(save_dir, model_dir)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir,f'classifier_val_acc{val_accuracy:.4f}.pth'))

    return model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train MLP Classifier')
    parser.add_argument('--input_dim', type=int, default=1000, help='Input dimension of the classifier (default: 1000)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension of the classifier (default: 512)')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer (default: 0.0001)')
    parser.add_argument('--data_paths', type=str, default="/home/dfl32/project/ifm/cinemaot_data/data_paths.json", help='Path to the JSON file containing dataset paths')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training (default: 100)')
    parser.add_argument('--pert_split', type=str, default="ct_pert", help='Perturbation split for the dataset (default: ct_pert)')
    parser.add_argument('--label_name', type=str, default="cell_type", help='Label name for the classifier')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for the optimizer (default: 0.0)')
    parser.add_argument('--use_wandb', action="store_true", help='Flag to use Weights and Biases for logging (default: True)')
    args = parser.parse_args()
    
    

    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    use_wandb = args.use_wandb
    label_name = args.label_name
    if use_wandb:
        wandb.init(project="IFM_SingleCell", name=f"MLP_Classifier_Train_{label_name}")
        wandb.config.update(args)

    with open(args.data_paths, 'r') as f:
        data_paths = json.load(f)

    dataset_path = data_paths[args.pert_split]
    train_dataset = load_from_disk(f"/gpfs/radev/scratch/dijk/sh2748/calmflow_singlecell/classifier_dataset/{args.pert_split}/processed_train_dataset")
    val_dataset = load_from_disk(f"/gpfs/radev/scratch/dijk/sh2748/calmflow_singlecell/classifier_dataset/{args.pert_split}/processed_val_dataset")

 
    
    # Train three MLP classifiers for three different labels
    if args.label_name == "cell_type":
        output_dim = 7
    elif args.label_name == "perturbation":
        output_dim = 10
    elif args.label_name == "chronicity":
        output_dim = 2
    else:
        raise NotImplementedError
    
    classifier = train_mlp_classifier(train_dataset, val_dataset, args.label_name, input_dim, hidden_dim, output_dim, args.pert_split, num_epochs, learning_rate, args.weight_decay, batch_size)