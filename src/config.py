import json

class Config():
    # -------------------------------------------------
    # Configuration: data_preprocessing.py
    # -------------------------------------------------

    # path to Chemical Component Dictionary dataset
    cif_path = "../data/components.cif"

    # make base datasets (on the basis of these dataframes, training and testing datasets can be created)
    make_bond_df = True  # if bond_df.csv is already computed in the /data folder, you can set this to False
    make_atom_df = True  # if atom_df.csv is already computed in the /data folder, you can set this to False
    make_atom_df_extended = True  # if atom_df_extended.csv is already computed in the /data folder, you can set this to False
    make_preprocessed_data = True  # if preprocessed_data.pkl is already computed in the /data folder, you can set this to False

    # loading and save paths for the dataframes
    atom_df_path = '../data/atom_df.csv'
    bond_df_path = '../data/bond_df.csv'
    atom_df_filtered_path = '../data/atom_df_extended.csv'
    preprocessed_data_save_path = '../data/preprocessed_data.pkl'

    # IMPORTANT PARAMETERS
    num_hydrogens = 1  # number of hydrogen atoms bonded to the central atom
    central_atom = 'C'  # central atom type 
    neighbor_depth = 1  # depth of neighbors to include in the feature vector; computation time increases for larger depts
    num_neighbors_to_centralatom = 4 # number of neighbors to central atom, this includes the hydrogen atom;

    # For Splitting the data
    train_test_split_ratio = 0.2
    random_state = 42

    # -------------------------------------------------
    # Configuration: train_models.py, model_selection.py, and test_models.py
    # -------------------------------------------------

    # for WandB
    use_wandb = True
    wandb_entity = "dl-coding"  # workspace that may contain many projects
    wandb_project_name = "hydrogen-prediction"  # project that is within the entity

    # for training and testing models
    datasets = ['dataset-C4-depth1', 'dataset-O2-depth1', 'dataset-C4-depth2', 'dataset-O2-depth2']
    alpha_values = [1, 0.1, 0.01, 0.001]
    do_polynomial_regression = True
    estimator_values = [100, 200, 500]
    n_jobs = -1  # number of CPUs to use for ensemble methods etc., '-1' means using all CPUs

    # -------------------------------------------------
    # Constants
    # -------------------------------------------------

    value_order_mapping = {'SING': 1, 'DOUB': 2, 'TRIP': 3}  # do not change 
    atomic_number_mapping = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16}  # do not change
    base_folder_name = f'dataset-{central_atom}{num_neighbors_to_centralatom}-depth{neighbor_depth}'  # do not change
    config_save_path = f'../data/{base_folder_name}/config.json'  # do not change
    dataset_save_path = f'../data/{base_folder_name}/dataset.pkl'  # do not change
    training_set_save_path = f'../data/{base_folder_name}/training-validation.npz'  # do not change
    testing_set_save_path = f'../data/{base_folder_name}/testing.npz'  # do not change
    connections_set_save_path = f'../data/{base_folder_name}/connections.pkl'  # do not change

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