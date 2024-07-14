import json

class Config():
    # -------------------------------------------------
    # Configuration: data_preprocessing2.py
    # -------------------------------------------------

    # path to Chemical Component Dictionary dataset
    cif_path = "../data/components.cif"

    # make base datasets (on the basis of these dataframes, training and testing datasets can be created)
    make_bond_df = False  # if bond_df.csv is already computed in the /data folder, you can set this to False
    make_atom_df = False  # if atom_df.csv is already computed in the /data folder, you can set this to False
    make_atom_df_extended = False  # if atom_df_extended.csv is already computed in the /data folder, you can set this to False
    make_preprocessed_data = False  # if preprocessed_data.pkl is already computed in the /data folder, you can set this to False

    # loading and save paths for the dataframes
    atom_df_path = '../data/atom_df.csv'
    bond_df_path = '../data/bond_df.csv'
    atom_df_filtered_path = '../data/atom_df_extended.csv'
    preprocessed_data_save_path = '../data/preprocessed_data.pkl'

    # IMPORTANT PARAMETERS
    num_hydrogens = 1  # number of hydrogen atoms bonded to the central atom
    central_atom = 'O'  # central atom type 
    neighbor_depth = 2  # depth of neighbors to include in the feature vector; computation time increases for larger depts
    num_neighbors_to_centralatom = 2 # number of neighbors to central atom, this includes the hydrogen atom;
    value_order_mapping = {'SING': 1, 'DOUB': 2, 'TRIP': 3}  # do not change this
    atomic_number_mapping = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16}  # do not change this

    # For Splitting the data
    train_test_split_ratio = 0.2
    random_state = 42

    # save paths
    base_folder_name = 'dataset-O2-depth2-withcon' # you only need to specify this
    config_save_path = f'../data/{base_folder_name}/config.json'
    dataset_save_path = f'../data/{base_folder_name}/dataset.pkl'
    training_set_save_path = f'../data/{base_folder_name}/training-validation.npz'
    testing_set_save_path = f'../data/{base_folder_name}/testing.npz'
    connections_set_save_path = f'../data/{base_folder_name}/connections.pkl'

    # For development (ignore)
    dev = False # let this False

    # -------------------------------------------------
    # Configuration: train_models.py
    # -------------------------------------------------

    # For WandB
    use_wandb = False
    wandb_project_name = "hydrogen-prediction"
    datasets = ['dataset-C4-depth1']  # ['dataset-C4-depth1', 'dataset-O2-depth1', 'dataset-C4-depth2', 'dataset-O2-depth2']
    alpha_values = [1, 0.1, 0.01, 0.001]
    do_polynomial_regression = True
    estimator_values = [100, 200, 500]
    n_jobs = -1  # number of CPUs to use for ensemble methods etc.

    # -------------------------------------------------
    # Class methods to be able to save the current config. py
    # -------------------------------------------------

    @classmethod
    def to_dict(cls):
        return {attr: getattr(cls, attr) for attr in dir(cls) if not callable(getattr(cls, attr)) and not attr.startswith("__")}
    
    @classmethod
    def save_to_json(cls, filepath):
        with open(filepath, 'w') as f:
            json.dump(cls.to_dict(), f, indent=4)