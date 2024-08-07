import logging
import os

import numpy as np
import umap
import matplotlib.pyplot as plt

logging.basicConfig(format='[%(levelname)s:%(asctime)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_umap(
    gt_data,
    gen_data,
    plot_name,
    save_dir="/home/dfl32/project/ifm/umaps/"
):
    
    # Combine the two datasets
    combined_data = np.vstack((gt_data, gen_data))

    # Fit and transform the combined data using UMAP
    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_embedding = umap_model.fit_transform(combined_data)

    # Split the transformed data back into the two original sets
    num_samples = gt_data.shape[0]
    data_dim = gt_data.shape[1]
    umap_gt = umap_embedding[:num_samples]
    umap_gen = umap_embedding[num_samples:]

    # Plot the results
    plt.figure(figsize=(5, 4))
    plt.scatter(umap_gt[:, 0], umap_gt[:, 1], color='blue', label='ground truth', alpha=0.5, s=0.5)
    plt.scatter(umap_gen[:, 0], umap_gen[:, 1], color='red', label='generated', alpha=0.5, s=0.5)
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.title(f'{data_dim} PCA Dimensions\n{num_samples} Cells')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    save_path = os.path.join(save_dir, plot_name)
    plt.savefig(save_path, bbox_inches='tight')
    logger.info(f"UMAP saved to {save_path}")