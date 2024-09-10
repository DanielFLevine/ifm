# This is not even necessary


import torch
from torch.utils.data import Dataset, DataLoader
import anndata as ad
from ipdb import set_trace
import numpy as np
import scanpy as sc

def convert_labels_to_classes(labels):
    # Convert labels to unique classes based on perturbation, cell type, and chronicity combinations.
    perturbations, cell_types, chronicities = labels["perturbations"], labels["cell_types"], labels["chronicities"]
    assert len(perturbations) == len(cell_types) == len(chronicities)
    
    unique_combinations = sorted(list({(perturbations[i], cell_types[i], chronicities[i]) for i in range(len(perturbations))})) # fix an ordering so we have consistent class labels
    class_map = {combination: i for i, combination in enumerate(unique_combinations)}
    
    classes = np.array([class_map[(perturbations[i], cell_types[i], chronicities[i])] for i in range(len(perturbations))])
    print(f"Number of unique classes: {len(unique_combinations)}")
    print("Unique combinations:")
    for combination in unique_combinations:
        print(combination)
    return classes, unique_combinations

class TestLabelDataset(Dataset):
    def __init__(self, adata): 
        super().__init__()
        
        self.dataset_data = {
            "perturbation": adata.obs["perturbation"].to_numpy(),
            "cell_type": adata.obs["cell_type0528"].to_numpy(),
            "chronicity": adata.obs["chronicity"].to_numpy(),
            "combined_labels": adata.obs["combined_labels"].to_numpy()
        }

    def __len__(self):
        return len(self.dataset_data["perturbation"])

    def __getitem__(self, idx):
        return {
            "perturbation": self.dataset_data["perturbation"][idx],
            "cell_type": self.dataset_data["cell_type"][idx],
            "chronicity": self.dataset_data["chronicity"][idx],
            "combined_labels": self.dataset_data["combined_labels"][idx]
        }
    
    def get_unique_label_combinations(self):
        unique_combinations = set(self.dataset_data["combined_labels"])
        return unique_combinations
    
    def get_indices_of_class(self, class_label):
        indices = np.where(self.dataset_data["combined_labels"] == class_label)[0]
        return indices
    
    @property
    def num_classes(self):
        return len(self.unique_combinations)

# # Example usage
if __name__ == "__main__":
    adata = sc.read_h5ad(f"/home/dfl32/project/ifm/cinemaot_data/conditional_cinemaot/ct_pert_split/test_data_ct_pert_split.h5ad") # shape (5902, 21710)
    n_cells_before_dropping = adata.shape[0]
    print("Dropping classes occurring less than 1%...")
    adata = drop_rare_classes(adata)
    n_cells_after_dropping = adata.shape[0]
    dataset = TestLabelDataset(adata)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Select one batch from the dataloader
    batch = next(iter(dataloader))
    # Print the shapes of all keys in the batch
    set_trace()