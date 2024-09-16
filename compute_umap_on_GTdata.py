# Compute a UMAP model on GT data and save it.
import scanpy as sc
import umap
import pickle
import os
from utils.metrics import transform_gpu
from scipy.sparse import issparse
import argparse
from utils.data_utils import drop_rare_classes
import logging
logging.basicConfig(format='[%(levelname)s:%(asctime)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

'''
    This file fits 2D UMAP models on GT data (1000D, after PCA)
'''

def parse_args():
    parser = argparse.ArgumentParser(description="Compute UMAP on GT data and save the model.")
    parser.add_argument('--pert_split', type=str, default="ct_pert")
    return parser.parse_args()


def compute_umap_on_GTdata(pert_split):
    # Load the AnnData object
    adata_path = f"/home/dfl32/project/ifm/cinemaot_data/conditional_cinemaot/{pert_split}_split/test_data_{pert_split}_split.h5ad"
    adata = sc.read_h5ad(adata_path)

    n_cells_before_dropping = adata.shape[0]
    logger.info("Dropping classes occurring less than 1%...")
    adata = drop_rare_classes(adata)
    n_cells_after_dropping = adata.shape[0]
    logger.info(f"Done dropping. Before dropping: {n_cells_before_dropping} cells, after dropping: {n_cells_after_dropping}")
    
    # Check if the data is sparse and convert to dense if necessary
    if issparse(adata.X):
        expression_data = adata.X.toarray()
    else:
        expression_data = adata.X
    
    # Initialize and fit the UMAP model
    # Load saved PCA model. 
    save_dir = "/home/dfl32/project/ifm/projections"
    save_name = f"{pert_split}_pcadim1000_numsamples10000.pickle"
    save_path = os.path.join(save_dir, save_name)
    with open(save_path, 'rb') as f:
        pca = pickle.load(f)
        logger.info("PCA model loaded!")

    logger.info("Transforming PCA ...")
    pca_expression_data = transform_gpu(expression_data, pca)
    logger.info("Done.")

    logger.info("Fitting UMAP ...")
    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_embedding = umap_model.fit_transform(pca_expression_data)
    logger.info("Done")


    umap_save_path = "/gpfs/radev/scratch/dijk/sh2748/calmflow_singlecell/umap_models/"
    umap_save_path = os.path.join(umap_save_path, f"{pert_split}_split_umap_model.pkl")
    
    # Save the UMAP model
    logger.info(f"Saving UMAP model to {umap_save_path}")
    with open(umap_save_path, 'wb') as f:
        pickle.dump(umap_model, f)
    logger.info("Done.")
    return umap_embedding

# Example usage
if __name__ == "__main__":
    args = parse_args()
    umap_embedding = compute_umap_on_GTdata(args.pert_split)
    logger.info(umap_embedding.shape)