CELL_TYPES = [
    'Plasma',
    'NK',
    'CD4 T',
    'Dendritic',
    'Monocyte',
    'CD8 T',
    'B'
]

PERTURBATIONS = [
    'IL-6',
    'IFNa2',
    'IFNb + IFNg',
    'IFNb + IL-6',
    'IFNg',
    'TNFa',
    'IFNb',
    'IFN-III',
    'IFNb + TNFa' 
]

def combo_split(combo_type: str):
    test_combos = set()
    if combo_type == 'fixed':
        perturb_idx = 0
        for i, cell_type in enumerate(CELL_TYPES):
            pert_1 = PERTURBATIONS[perturb_idx]
            pert_2 = PERTURBATIONS[(perturb_idx+1)%len(PERTURBATIONS)]
            test_combos.add((cell_type, pert_1, 'chronic'))
            test_combos.add((cell_type, pert_2, 'acute'))
            perturb_idx = (perturb_idx + 2) % len(PERTURBATIONS)
    elif combo_type == 'chronic_acute':
        perturb_idx = 0
        for i, cell_type in enumerate(CELL_TYPES):
            pert_1 = PERTURBATIONS[perturb_idx]
            pert_2 = PERTURBATIONS[(perturb_idx+1)%len(PERTURBATIONS)]
            test_combos.add((cell_type, pert_1, 'chronic'))
            test_combos.add((cell_type, pert_1, 'acute'))
            test_combos.add((cell_type, pert_2, 'chronic'))
            test_combos.add((cell_type, pert_2, 'acute'))
            perturb_idx = (perturb_idx + 2) % len(PERTURBATIONS)
    else:
        raise ValueError(f"Combination split {combo_type} is not supported.")
    return test_combos

def combo_split_nochron():
    test_combos = set()
    perturb_idx = 0
    for i, cell_type in enumerate(CELL_TYPES):
        pert_1 = PERTURBATIONS[perturb_idx]
        pert_2 = PERTURBATIONS[(perturb_idx+1)%len(PERTURBATIONS)]
        test_combos.add((cell_type, pert_1))
        test_combos.add((cell_type, pert_2))
        perturb_idx = (perturb_idx + 2) % len(PERTURBATIONS)
    return test_combos
        