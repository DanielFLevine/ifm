import pandas as pd
from ipdb import set_trace

def drop_rare_classes(adata, min_percentage=1):
    """
    This function drops rare classes from the AnnData object based on a minimum percentage threshold.
    
    Parameters:
    adata (AnnData): The annotated data matrix.
    min_percentage (float): The minimum percentage threshold for a class to be retained. Classes with a percentage lower than this threshold will be dropped. Default is 1.
    
    Returns:
    AnnData: The filtered annotated data matrix with rare classes removed.
    """
    adata.obs['combined_labels'] = adata.obs.apply(lambda row: f"{row['cell_type0528']}_{row['perturbation']}_{row['chronicity']}", axis=1)
    label_counts = adata.obs['combined_labels'].value_counts(normalize=True) * 100
    rare_labels = label_counts[label_counts < min_percentage].index
    adata = adata[~adata.obs['combined_labels'].isin(rare_labels)]
    return adata

def get_control_adata(adata, class_label):
    """
    This function retrieves control data from the AnnData object based on the given class label.
    
    Parameters:
    adata (AnnData): The annotated data matrix.
    class_label (str): The class label in the format '{cell_type}+{perturbation}+{chronicity}'.
    
    Returns:
    AnnData: A AnnData object containing the control data where the perturbation is 'No stimulation'.
    """
    # Split the class_label into cell_type_label and chronicity_label
    # class_label is expected to be in the format 'cell_type+perturbation+chronicity'
    if not len(class_label.split('_')) == 3:
        set_trace()
    cell_type_label, _, chronicity_label = class_label.split('_')
    control_data = adata[(adata.obs['cell_type0528'] == cell_type_label) & 
                            (adata.obs['chronicity'] == chronicity_label) & 
                            (adata.obs['perturbation'] == 'No stimulation')]
    return control_data

def get_control_adata_scgen(adata, class_label):
    """
    This function retrieves control data from the AnnData object based on the given class label.
    
    Parameters:
    adata (AnnData): The annotated data matrix.
    class_label (str): The class label in the format '{cell_type}+{perturbation}+{chronicity}'.
    
    Returns:
    AnnData: A AnnData object containing the control data where the perturbation is 'No stimulation'.
    """
    # Split the class_label into cell_type_label and chronicity_label
    # class_label is expected to be in the format 'cell_type+perturbation+chronicity'
    if not len(class_label.split('_')) == 3:
        set_trace()
    cell_type_label, _, chronicity_label = class_label.split('_')
    control_data = adata[(adata.obs['cell_type'] == cell_type_label) & 
                            (adata.obs['chronicity'] == chronicity_label) & 
                            (adata.obs['perturbation'] == 'No stimulation')]
    return control_data