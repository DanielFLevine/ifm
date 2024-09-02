import logging
import os

from tqdm import tqdm

import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from ipdb import set_trace
import seaborn as sns

logging.basicConfig(format='[%(levelname)s:%(asctime)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_umap(
    gt_data,
    gen_data,
    plot_name,
    labels: dict,
    save_dir="/home/sh2748/ifm/umaps/"
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
    umap_name = "_total_umap.png"
    plt.figure(figsize=(5, 4))
    plt.scatter(umap_gt[:, 0], umap_gt[:, 1], color='blue', label='ground truth', alpha=0.5, s=0.5)
    plt.scatter(umap_gen[:, 0], umap_gen[:, 1], color='red', label='generated', alpha=0.5, s=0.5)
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.title(f'{data_dim} PCA Dimensions\n{num_samples} Cells')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    save_path = os.path.join(save_dir, plot_name+umap_name)
    plt.savefig(save_path, bbox_inches='tight')
    logger.info(f"UMAP saved to {save_path}")
    plt.close()

    # set_trace()
    for keys in labels:
        # Plot the results with color according to the label
        umap_name=f"_GT_{keys}.png"
        # label_encoder = LabelEncoder()
        # numeric_labels = label_encoder.fit_transform(labels[keys])
        plt.figure(figsize=(10, 8))
        # plt.scatter(umap_gt[:, 0], umap_gt[:, 1], c=numeric_labels, cmap='viridis', alpha=0.5, s=0.5)
        # handles = [plt.Line2D([0], [0], marker='o', color='w', label=label_encoder.classes_[i],
        #               markersize=10, markerfacecolor=plt.cm.viridis(i / len(label_encoder.classes_)))
        #    for i in range(len(label_encoder.classes_))]
        # plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
        sns.scatterplot(x=umap_gt[:, 0], y=umap_gt[:, 1], hue=labels[keys], alpha=0.7, s=4)
        # plt.xticks([])
        # plt.yticks([])
        plt.title(f'UMAP of GT data colored by {keys}')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        save_path = os.path.join(save_dir, plot_name+umap_name)
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"UMAP saved to {save_path}")
        plt.close()
        




        umap_name=f"_generated_{keys}.png"
        plt.figure(figsize=(10, 8))
        # plt.scatter(umap_gen[:, 0], umap_gen[:, 1], c=numeric_labels, cmap='viridis', alpha=0.5, s=0.5)
        # handles = [plt.Line2D([0], [0], marker='o', color='w', label=label_encoder.classes_[i],
        #               markersize=10, markerfacecolor=plt.cm.viridis(i / len(label_encoder.classes_)))
        #    for i in range(len(label_encoder.classes_))]
        # plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
        sns.scatterplot(x=umap_gen[:, 0], y=umap_gen[:, 1], hue=labels[keys], alpha=0.7, s=4)
        # plt.xticks([])
        # plt.yticks([])
        plt.title(f'UMAP of IFM generated data colored by {keys}')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        save_path = os.path.join(save_dir, plot_name+umap_name)
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"UMAP saved to {save_path}")
        plt.close()

def plot_umap_after_train(
    gt_pca,
    model,
    time_points,
    input_dim,
    space_dim,
    device,
    temp=1.0,
    num_samples=5000,
    batch_size=100,
    idfm=False
):
    model.eval()
    num_steps = num_samples//batch_size

    euler_step_size = 1/(time_points-1)
    all_trajs = []
    with torch.no_grad():
        cells = []
        for step in tqdm(range(num_steps)):
            inputs = torch.normal(0.0, 1.0, size=(batch_size, 1, input_dim)).to(device)
            for _ in range(time_points-1):
                outputs = model.cell_enc(inputs)

                # Reshape for spatial integration
                batch_size, seq_len, feature = outputs.shape
                outputs = outputs.view(batch_size, seq_len, space_dim, feature // space_dim)
                outputs = outputs.view(batch_size, seq_len* space_dim, feature // space_dim)


                outputs = model.gpt_neox(inputs_embeds=outputs).last_hidden_state

                outputs, _, _ = model.cell_dec(outputs, temperature=temp)
                last_outputs = outputs[:, -1:, :]
                if idfm:
                    last_outputs = inputs[:, -1:, :] + (euler_step_size * last_outputs)
                inputs = torch.concat([inputs, last_outputs], axis=1)
            cells.append(outputs[:, -1, :].detach().cpu().numpy())
            all_trajs.append(inputs.detach().cpu().numpy())
        cells = np.concatenate(cells, axis=0)
        all_trajs = np.concatenate(all_trajs, axis=0)
    
    # Combine the two datasets
    combined_data = np.vstack((gt_pca, cells))

    # Fit and transform the combined data using UMAP
    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_embedding = umap_model.fit_transform(combined_data)

    # Split the transformed data back into the two original sets
    umap_gt_pca = umap_embedding[:num_samples]
    umap_cells = umap_embedding[num_samples:]

    # Plot the results
    plt.figure(figsize=(5, 4))
    plt.scatter(umap_gt_pca[:, 0], umap_gt_pca[:, 1], color='blue', label='ground truth', alpha=0.5, s=0.5)
    plt.scatter(umap_cells[:, 0], umap_cells[:, 1], color='red', label='generated', alpha=0.5, s=0.5)
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.title(f'{time_points} Time Points\n{input_dim} PCA Dimensions\n{num_samples} Cells')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()