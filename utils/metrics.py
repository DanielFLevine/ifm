import logging

import numpy as np
import ot
import torch
from torch import nn
import torch.nn.functional as F
import scanpy as sc
import umap
from sklearn.metrics import pairwise, pairwise_distances
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
import os

logging.basicConfig(format='[%(levelname)s:%(asctime)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

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

def energy_distance(X, Y):
    """Compute the Energy Distance between two distributions X and Y.

    Parameters:
    X : numpy array of shape (n_samples_X, n_features)
    Y : numpy array of shape (n_samples_Y, n_features)

    Returns:
    energy_dist : float
    """
    # Pairwise distances within X
    XX_dist = cdist(X, X, metric='euclidean')
    # Pairwise distances within Y
    YY_dist = cdist(Y, Y, metric='euclidean')
    # Pairwise distances between X and Y
    XY_dist = cdist(X, Y, metric='euclidean')
    
    # Compute mean distances
    XX_mean = XX_dist.mean()
    YY_mean = YY_dist.mean()
    XY_mean = XY_dist.mean()

    # Energy distance formula
    energy_dist = 2 * XY_mean - XX_mean - YY_mean
    
    return energy_dist


def compute_wass(X, Y, reg=0.01):
    # Compute the cost matrix (squared Euclidean distances)
    logger.info("Computing distance matrix for 2-Wasserstein...")
    M = ot.dist(X, Y, metric='sqeuclidean')

    # Normalize the cost matrix
    M /= M.max()

    # Assume uniform distribution of weights
    a, b = np.ones((X.shape[0],)) / X.shape[0], np.ones((Y.shape[0],)) / Y.shape[0]

    logger.info("Computing 2-Wasserstein distance...")
    wasserstein_dist = ot.sinkhorn2(a, b, M, reg=reg)
    return wasserstein_dist

def transform_gpu(data, pca):
    components = pca.components_
    mean = pca.mean_

    data_gpu = torch.tensor(data, device='cuda')
    components_gpu = torch.tensor(components, device='cuda')
    mean_gpu = torch.tensor(mean, dtype=torch.float32).cuda()

    # Center the data by subtracting the mean
    data_centered = data_gpu - mean_gpu
    
    # Transform the data using the PCA components
    transformed_data_gpu = torch.matmul(data_centered, components_gpu.T)
    
    transformed_data = transformed_data_gpu.cpu().numpy()
    return transformed_data

def umap_embed(
    gt_data,
    gen_data,
):
    
    # Combine the two datasets
    combined_data = np.vstack((gt_data, gen_data))

    # Fit and transform the combined data using UMAP
    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_embedding = umap_model.fit_transform(combined_data)

    # Split the transformed data back into the two original sets
    num_samples = gt_data.shape[0]
    umap_gt = umap_embedding[:num_samples]
    umap_gen = umap_embedding[num_samples:]

    return umap_gt, umap_gen

def total_variation_distance(p, q):
    """
    Compute the total variation distance between two discrete probability distributions.
    """
    return 0.5 * np.sum(np.abs(p - q))

def weighted_total_variation_distance(p, q, weights):
    """
    Compute the weighted total variation distance between two discrete probability distributions.
    
    Parameters:
    p : numpy array of shape (n_classes,)
        Ground truth probability distribution.
    q : numpy array of shape (n_classes,)
        Generated probability distribution.
    weights : numpy array of shape (n_classes,)
        Weights for each class, typically the ground truth probabilities.
    
    Returns:
    weighted_tvd : float
        Weighted total variation distance.
    """
    return 0.5 * np.sum(weights * np.abs(p - q))

def evaluate_model(checkpoint_path='/home/dfl32/scratch/unconditional_classifier_combined_labels/checkpoints_hidden_dim_256_2/checkpoint_step100000.pth', 
                   generated_data=None, 
                   adata_path='/home/dfl32/project/ifm/cinemaot_data/conditional_cinemaot/integrated_ifm_leiden_02.h5ad', 
                   batch_size=64):
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

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adata = sc.read_h5ad(adata_path)

    # Load the model
    input_dim = 1000
    hidden_dim = 256  # Adjust based on your model
    output_dim = adata.obs['leiden'].nunique()
    model = SimpleMLP(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Convert generated_data to numpy array if it's not already
    if isinstance(generated_data, torch.Tensor):
        generated_data = generated_data.cpu().numpy()
    elif not isinstance(generated_data, np.ndarray):
        generated_data = np.array(generated_data)

    # Ensure generated_data is 2D
    if generated_data.ndim == 1:
        generated_data = generated_data.reshape(-1, 1)

    # Convert the generated data to a tensor and move to device
    generated_data = torch.tensor(generated_data, dtype=torch.float32).to(device)

    # Create DataLoader for batching
    dataset = TensorDataset(generated_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Compute the average entropy of the generated samples
    all_probs = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch[0].to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)

    all_probs = torch.cat(all_probs, dim=0)
    incep_score = inception_score(all_probs)
    sample_entropy = -torch.sum(all_probs * torch.log(all_probs), dim=1)
    avg_entropy = torch.mean(sample_entropy).item()

    # Compute plausibility score
    plausibility_score = torch.mean(torch.max(all_probs, dim=1)[0]).item()

    # Compute the average probability vector
    avg_prob_vector = torch.mean(all_probs, dim=0)

    # Compute the true probabilities
    true_labels = adata.obs['leiden'].astype('category').cat.codes.values
    true_labels = torch.tensor(true_labels, dtype=torch.int64).to(device)  # Ensure true_labels are integers
    true_probs = torch.tensor([torch.mean((true_labels == i).float()) for i in range(output_dim)]).to(device)

    # Compute the KL divergence between the probabilities
    kl_div = F.kl_div(avg_prob_vector.log(), true_probs, reduction='batchmean').item()

    # Calculate total variation distance
    tv_distance = 0.5 * torch.sum(torch.abs(avg_prob_vector - true_probs)).item()

    # Compute weighted total variation distance
    weighted_tv_distance = weighted_total_variation_distance(avg_prob_vector.cpu().numpy(), true_probs.cpu().numpy(), true_probs.cpu().numpy())

    # Compute KL divergence between predicted and true label distributions
    predicted_labels = torch.argmax(all_probs, dim=1).cpu().numpy()
    predicted_label_dist = np.bincount(predicted_labels, minlength=output_dim) / len(predicted_labels)
    logger.info(f"Predicted label distribution: {dict(enumerate(predicted_label_dist))}")
    
    # Cast true_labels to integers before using np.bincount
    true_labels_int = true_labels.cpu().numpy().astype(int)
    true_label_dist = np.bincount(true_labels_int, minlength=output_dim) / len(true_labels)
    
    label_kl_div = entropy(true_label_dist, predicted_label_dist)

    return avg_entropy, kl_div, tv_distance, label_kl_div, weighted_tv_distance, incep_score, plausibility_score

def inception_score(p_yx):
    """
    Compute the Inception Score for the generated data.

    Parameters:
    p_yx : torch tensor of shape [n_samples, n_classes)]
        The predicted class probabilities for the generated data.
        p(y|x), where y is the label, x is the data

    Returns:
    inception_score : float
        The computed Inception Score.
    """
    py = p_yx.mean(dim=0) # marginal prob, p(y)
    kl_div = torch.distributions.kl.kl_divergence(
        torch.distributions.Categorical(probs=p_yx),
        torch.distributions.Categorical(probs=py)
    ).mean(dim=0)
    
    inception_score = kl_div.exp()
    return inception_score.item()

def leiden_KL(gt_labels, pred_labels):
    """
    Compute the KL divergence between the ground truth and generated data using Leiden clustering.


    Returns:
    kl_divergence : torch.Tensor
        The computed KL divergence between the ground truth and generated data.
    """

    # We use the Leiden labels as bins directly
    gt_hist = np.bincount(gt_labels) / len(gt_labels)
    pred_hist = np.bincount(pred_labels, minlength=len(gt_hist)) / len(pred_labels)

    gt_hist_tensor = torch.tensor(gt_hist)
    pred_hist_tensor = torch.tensor(pred_hist)

    # set_trace()

    kl_divergence = torch.distributions.kl.kl_divergence(
        torch.distributions.Categorical(probs=gt_hist_tensor.view(-1)),
        torch.distributions.Categorical(probs=pred_hist_tensor.view(-1))
    )
    # set_trace()
    return kl_divergence.item()

def adaptive_rbf_kernel(X, Y=None, k=5):
    """
    Adaptive RBF kernel that adjusts the bandwidth based on local densities.
    
    Arguments:
        X {[n_samples1, dim]} -- [Input matrix X]
        Y {[n_samples2, dim]} -- [Optional input matrix Y; if None, uses X for both]
        k {int} -- [Number of neighbors to determine the adaptive bandwidth] (default: {5})
    
    Returns:
        Kernel matrix K using adaptive RBF kernel
    """
    if Y is None:
        Y = X
    
    # Compute pairwise distances (using euclidean for nearest neighbor bandwidth calculation)
    dists_X = pairwise_distances(X, X, metric='euclidean')
    dists_Y = pairwise_distances(Y, Y, metric='euclidean') if Y is not X else dists_X
    
    # Compute adaptive bandwidth for each point based on k-nearest neighbors
    sigma_X = np.mean(np.sort(dists_X, axis=1)[:, 1:k+1], axis=1)  # Use k nearest neighbors
    sigma_Y = np.mean(np.sort(dists_Y, axis=1)[:, 1:k+1], axis=1)  # Use k nearest neighbors

    # Compute squared pairwise distances directly
    sq_dists = pairwise_distances(X, X, metric='sqeuclidean') if Y is X else \
               pairwise_distances(X, Y, metric='sqeuclidean')
    
    # Compute the adaptive RBF kernel matrix with squared distances and 2 * sigma^2 in the denominator
    K = np.exp(-sq_dists / (2 * (sigma_X[:, None] * sigma_X[None, :]))) if Y is X else \
        np.exp(-sq_dists / (2 * (sigma_X[:, None] * sigma_Y[None, :])))
    
    return K

def mmd_rbf_adaptive(X, Y, k=5):
    """
    MMD using adaptive RBF (Gaussian) kernel.
    
    Arguments:
        X {[n_sample1, dim]} -- [Input matrix X]
        Y {[n_sample2, dim]} -- [Input matrix Y]
    
    Keyword Arguments:
        k {int} -- [Number of neighbors to use for adaptive bandwidth] (default: {5})
    
    Returns:
        [scalar] -- [MMD value]
    """
    XX = adaptive_rbf_kernel(X, X, k)
    YY = adaptive_rbf_kernel(Y, Y, k)
    XY = adaptive_rbf_kernel(X, Y, k)
    
    return XX.mean() + YY.mean() - 2 * XY.mean()

def generate_latex_table(all_metrics, model_name, split_type):
    """
    Generate a LaTeX table with models as rows and metrics as columns.
    
    Args:
    all_metrics (dict): A dictionary with model identifiers as keys and metric dictionaries as values.
    
    Returns:
    None. Saves the LaTeX table to a file.
    """
    table_content = []
    table_content.append("\\documentclass{article}")
    table_content.append("\\usepackage{booktabs}")
    table_content.append("\\usepackage{longtable}")
    table_content.append("\\usepackage{array}")
    table_content.append("\\begin{document}")
    table_content.append("\\begin{longtable}{l*{20}{c}}")  # Adjust the number of columns as needed
    table_content.append("\\caption{Metrics for all models} \\\\")
    table_content.append("\\toprule")

    # Get all unique metric names
    all_metric_names = set()
    for metrics in all_metrics.values():
        all_metric_names.update(metrics.keys())
    all_metric_names = sorted(list(all_metric_names))

    # Replace "R^2" with "$R^2$" in metric names
    all_metric_names = [name.replace("R^2", "$R^2$") for name in all_metric_names]

    # Add header row
    header = ["Model"] + all_metric_names
    table_content.append(" & ".join(header) + " \\\\")
    table_content.append("\\midrule")
    table_content.append("\\endfirsthead")
    
    # Repeat header for subsequent pages
    table_content.append("\\multicolumn{" + str(len(header)) + "}{c}{\\tablename\\ \\thetable\\ -- continued from previous page} \\\\")
    table_content.append("\\toprule")
    table_content.append(" & ".join(header) + " \\\\")
    table_content.append("\\midrule")
    table_content.append("\\endhead")
    
    table_content.append("\\midrule")
    table_content.append("\\multicolumn{" + str(len(header)) + "}{r}{Continued on next page} \\\\")
    table_content.append("\\endfoot")
    
    table_content.append("\\bottomrule")
    table_content.append("\\endlastfoot")
    
    # Add data rows
    for model, metrics in all_metrics.items():
        row = [model]
        for metric_name in all_metric_names:
            original_metric_name = metric_name.replace("$R^2$", "R^2")
            if original_metric_name in metrics:
                mean = np.mean(metrics[original_metric_name])
                std = np.std(metrics[original_metric_name])
                row.append(f"{mean:.5f} $\\pm$ {std:.5f}")
            else:
                row.append("-")
        table_content.append(" & ".join(row) + " \\\\")
    
    table_content.append("\\end{longtable}")
    table_content.append("\\end{document}")
    
    # Join all lines and add newline characters
    table_content = "\n".join(table_content)
    
    # Save the table to a file
    save_dir = "/home/dfl32/project/ifm/tables"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{model_name}_{split_type}_metrics_table_all_models.tex")
    
    with open(file_path, "w") as f:
        f.write(table_content)
    
    logger.info(f"LaTeX table saved to {file_path}")

