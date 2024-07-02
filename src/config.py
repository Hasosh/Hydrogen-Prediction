class Config():
    # make base datasets (on the basis of these dataframes, training and testing datasets can be created)
    make_bond_df = False
    make_atom_df = False
    make_atom_df_extended = False

    num_hydrogens = 1
    central_atom = 'C'
    allowed_atomic_numbers = [1, 6, 8, 7, 16]  # ['H', 'C', 'O', 'N', 'S']
    max_neighbor_depth = 1  # not advised to use larger number than 2 because not implemented memory efficient (big joins are made)
    num_neighbors_to_centralatom = 4 # this includes the hydrogen atom; IMPORTANT: currently only works for max_neighbor_depth=1

    train_test_split_ratio = 0.2
    random_state = 42

    cif_path = "../data/components.cif"
    atom_df_path = '../data/atom_df.csv'
    bond_df_path = '../data/bond_df.csv'
    atom_df_filtered_path = '../data/atom_df_extended.csv'

    dataset_save_path = '../data/centralatom-C_numhydrogens-1_numneighbors-3_depth-1/dataset.pkl'
    training_set_save_path = '../data/centralatom-C_numhydrogens-1_numneighbors-3_depth-1/training-validation.pkl'
    testing_set_save_path = '../data/centralatom-C_numhydrogens-1_numneighbors-3_depth-1/testing.pkl'
    max_length_save_path = '../data/centralatom-C_numhydrogens-1_numneighbors-3_depth-1/max_length.txt'

    dev = False # let this False