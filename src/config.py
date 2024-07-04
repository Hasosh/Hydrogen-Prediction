import json

class Config():
    # path to Chemical Component Dictionary dataset
    cif_path = "../data/components.cif"

    # make base datasets (on the basis of these dataframes, training and testing datasets can be created)
    make_bond_df = False
    make_atom_df = False
    make_atom_df_extended = False
    make_preprocessed_data = False

    # loading and save paths for the dataframes
    atom_df_path = '../data/atom_df.csv'
    bond_df_path = '../data/bond_df.csv'
    atom_df_filtered_path = '../data/atom_df_extended.csv'
    preprocessed_data_save_path = '../data/preprocessed_data.pkl'

    # IMPORTANT PARAMETERS
    num_hydrogens = 1
    central_atom = 'O'
    neighbor_depth = 2  # computation time increases for larger depts
    num_neighbors_to_centralatom = 2 # number of neighbors to central atom, this includes the hydrogen atom;
    value_order_mapping = {'SING': 1, 'DOUB': 2, 'TRIP': 3}  # do not change this
    atomic_number_mapping = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16}  # do not change this

    # For Splitting the data
    train_test_split_ratio = 0.2
    random_state = 42

    # save paths
    base_folder_name = 'dataset-O2-depth2' # you only need to specify this
    config_save_path = f'../data/{base_folder_name}/config.json'
    dataset_save_path = f'../data/{base_folder_name}/dataset.pkl'
    training_set_save_path = f'../data/{base_folder_name}/training-validation.npz'
    testing_set_save_path = f'../data/{base_folder_name}/testing.npz'

    # For development (ignore)
    dev = True # let this False

    @classmethod
    def to_dict(cls):
        return {attr: getattr(cls, attr) for attr in dir(cls) if not callable(getattr(cls, attr)) and not attr.startswith("__")}
    
    @classmethod
    def save_to_json(cls, filepath):
        with open(filepath, 'w') as f:
            json.dump(cls.to_dict(), f, indent=4)