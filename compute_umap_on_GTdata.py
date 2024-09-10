# Compute a UMAP model on GT data and save it.
import scanpy as sc
import umap
import pickle
import os
from utils.metrics import transform_gpu
from scipy.sparse import issparse

def compute_umap_on_GTdata(adata_path, umap_save_path):
    # Load the AnnData object
    adata = sc.read_h5ad(adata_path)
    
    # Check if the data is sparse and convert to dense if necessary
    if issparse(adata.X):
        expression_data = adata.X.toarray()
    else:
        expression_data = adata.X
    
    # Initialize and fit the UMAP model
    # Load saved PCA model. 
    save_dir = "/home/dfl32/project/ifm/projections"
    save_name = f"chron_pcadim1000_numsamples10000.pickle"
    save_path = os.path.join(save_dir, save_name)
    with open(save_path, 'rb') as f:
        pca = pickle.load(f)

    print("PCA model loaded!")
    print("Transforming PCA ...")
    pca_expression_data = transform_gpu(expression_data, pca)
    print("Done.")
    print("Fitting UMAP ...")
    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_embedding = umap_model.fit_transform(pca_expression_data)
    print("Done")
    
    # Save the UMAP model
    print(f"Saving UMAP model to {umap_save_path}")
    with open(umap_save_path, 'wb') as f:
        pickle.dump(umap_model, f)
    print("Done.")
    return umap_embedding

# Example usage
adata_path = "/home/dfl32/project/ifm/cinemaot_data/conditional_cinemaot/chron_split/test_data_chron_split.h5ad"
save_path = "/gpfs/radev/scratch/dijk/sh2748/calmflow_singlecell/umap_models/chron_split_umap_model.pkl"
umap_embedding = compute_umap_on_GTdata(adata_path, save_path)
