from gemmi import cif
import gemmi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import Config
from collections import defaultdict
from dscribe.descriptors import CoulombMatrix
from ase import Atoms
from sklearn.model_selection import train_test_split
import pickle
import os


def create_bond_df(doc):
    bond_site_cols =  ['comp_id', 'atom_id_1', 'atom_id_2', 'value_order']

    # List to collect data
    bond_data = []
    
    # Iterate over each block in the document
    for blk in tqdm(doc):
        # Find the bond information in the current block
        _chem_comp_bond = blk.find('_chem_comp_bond.', bond_site_cols)
        
        # Extract data directly into a list of tuples
        bond_data.extend([tuple(row) for row in _chem_comp_bond])
    
    # Create a DataFrame from the collected data
    bond_df = pd.DataFrame(bond_data, columns=bond_site_cols)

    # Save the consolidated DataFrame to a CSV file
    bond_df.to_csv(Config.bond_df_path, index=False)

    print(bond_df.head(10))


def create_atom_df(doc):
    # Define the columns to extract from the CIF block
    atom_site_cols =  ['comp_id', 'atom_id', 'type_symbol', 'model_Cartn_x', 'model_Cartn_y', 'model_Cartn_z']

    # List to collect data
    atom_site_data = []
    
    # Iterate over each block in the document
    for blk in tqdm(doc):
        # Find the atom site information in the current block
        _atom_site = blk.find('_chem_comp_atom.', atom_site_cols)
        
        # Extract data directly into a list of tuples
        atom_site_data.extend([tuple(row) for row in _atom_site])
    
    # Create a DataFrame from the collected data
    atom_df = pd.DataFrame(atom_site_data, columns=atom_site_cols)
    
    # Add a new column 'is_hydrogen' to indicate if the atom is hydrogen
    atom_df['is_hydrogen'] = atom_df['type_symbol'] == 'H'

    # Save the consolidated DataFrame to a CSV file
    atom_df.to_csv(Config.atom_df_path, index=False)

    print(atom_df.head(10))


""" 
atom_df_extended has one more column "bonded_hydrogens" that tells the number of hydrogens bonded to a certain atom
"""
def create_atom_df_extended(atom_df, bond_df):
    filtered_atom_df = atom_df[atom_df['is_hydrogen']]

    # Merge bond_df with atom_df on (comp_id, atom_id_1)
    merged_1 = pd.merge(bond_df, filtered_atom_df, left_on=['comp_id', 'atom_id_1'], right_on=['comp_id', 'atom_id'], suffixes=('_bond', '_atom'))
    
    # Merge bond_df with atom_df on (comp_id, atom_id_2)
    merged_2 = pd.merge(bond_df, filtered_atom_df, left_on=['comp_id', 'atom_id_2'], right_on=['comp_id', 'atom_id'], suffixes=('_bond', '_atom'))

    # Combine both bond directions
    bonded_hydrogens = pd.concat([
        merged_1[['comp_id', 'atom_id_2']].rename(columns={'atom_id_2': 'atom_id'}),
        merged_2[['comp_id', 'atom_id_1']].rename(columns={'atom_id_1': 'atom_id'})
    ])

    # Group by the two columns and count the occurrences
    grouped_counts = bonded_hydrogens.groupby(['comp_id', 'atom_id']).size().reset_index(name='counts')

    # Initialize the 'bonded_hydrogens' column to 0
    atom_df['bonded_hydrogens'] = 0

    # Merge atom_df with grouped_counts on 'comp_id' and 'atom_id'
    merged_df = pd.merge(atom_df, grouped_counts, on=['comp_id', 'atom_id'], how='left')

    # Update the 'bonded_hydrogens' column with the counts from the merge
    atom_df['bonded_hydrogens'] = merged_df['counts'].fillna(0).astype(int)

    # Save the updated atom_df to CSV
    atom_df.to_csv(Config.atom_df_filtered_path, index=False)

    print(atom_df.head(10))


def clean_dataframe(atom_df, bond_df):
    def remove_nan_rows(atom_df, bond_df):
        # Step 1: Identify rows with NaN comp_id and collect comp_ids to delete
        nan_comp_ids = set()

        # Find NaN comp_ids in atom_df
        nan_atom_rows = atom_df[atom_df.isna().any(axis=1)]
        nan_comp_ids.update(nan_atom_rows['comp_id'])

        # Find NaN comp_ids in bond_df
        nan_bond_rows = bond_df[bond_df.isna().any(axis=1)]
        nan_comp_ids.update(nan_bond_rows['comp_id'])

        # Convert comp_id to set for efficient lookup
        nan_comp_ids = set(nan_comp_ids)

        # Find all unique comp_ids to delete from atom_df and bond_df
        comp_ids_to_delete = set(atom_df[atom_df['comp_id'].isin(nan_comp_ids)]['comp_id']).union(
                             set(bond_df[bond_df['comp_id'].isin(nan_comp_ids)]['comp_id']))

        # Delete rows with identified comp_ids from both dataframes
        atom_df = atom_df[~atom_df['comp_id'].isin(comp_ids_to_delete)]
        bond_df = bond_df[~bond_df['comp_id'].isin(comp_ids_to_delete)]

        return atom_df, bond_df, comp_ids_to_delete

    def remove_question_mark_rows(atom_df, bond_df):
        # Identify rows in atom_df that contain "?" in any column
        rows_with_question_mark = atom_df.isin(["?"]).any(axis=1)

        # Collect the comp_ids from these rows
        comp_ids_to_delete = atom_df.loc[rows_with_question_mark, 'comp_id'].unique()

        # Remove all rows with these comp_ids from both dataframes
        atom_df_cleaned = atom_df[~atom_df['comp_id'].isin(comp_ids_to_delete)]
        bond_df_cleaned = bond_df[~bond_df['comp_id'].isin(comp_ids_to_delete)]

        return atom_df_cleaned, bond_df_cleaned, comp_ids_to_delete

    # Initial row counts
    initial_atom_row_count = len(atom_df)
    initial_bond_row_count = len(bond_df)

    # Remove NaN rows
    atom_df, bond_df, nan_comp_ids_to_delete = remove_nan_rows(atom_df, bond_df)

    # Remove "?" rows
    atom_df, bond_df, question_mark_comp_ids_to_delete = remove_question_mark_rows(atom_df, bond_df)

    # Combine comp_ids_to_delete from both steps
    #total_comp_ids_to_delete = nan_comp_ids_to_delete.union(question_mark_comp_ids_to_delete)

    # Final row counts
    final_atom_row_count = len(atom_df)
    final_bond_row_count = len(bond_df)

    # Print results
    print(f"Identified {len(nan_comp_ids_to_delete)} unique comp_ids with NaN values.")
    print(f"Identified {len(question_mark_comp_ids_to_delete)} unique comp_ids with '?' values in atom_df.")
    print(f"Deleted {initial_atom_row_count - final_atom_row_count} rows from atom_df.")
    print(f"Deleted {initial_bond_row_count - final_bond_row_count} rows from bond_df.")
    
    return atom_df, bond_df


# Join on atom_df to get coordinates of the current df depth
def join_coordinates(result_df, atom_df, join_number):
    # Perform the join
    merged_df = result_df.merge(
        atom_df,
        left_on=['comp_id', f'atom_id_joined{join_number}'],
        right_on=['comp_id', 'atom_id'],
        suffixes=('', f'_coor_joined{join_number}')
    )

    merged_df.drop(columns=[f'atom_id_coor_joined{join_number}'], inplace=True)

    return merged_df

# Join on bond_df to get bondings with the current df depth
def join_bonds(result_df, bond_df, join_number=1):
    join_column = 'atom_id' if join_number == 1 else f'atom_id_joined{join_number - 1}'

    # Merge on comp_id and atom_id_1
    merged_1 = result_df.merge(
        bond_df,
        left_on=['comp_id', join_column],
        right_on=['comp_id', 'atom_id_1'],
        suffixes=('', f'_bond{join_number}')
    )

    # Rename atom_id_2 to atom_id_joined and drop atom_id_1
    merged_1.rename(columns={'atom_id_2': f'atom_id_joined{join_number}'}, inplace=True)
    merged_1.drop(columns=['atom_id_1'], inplace=True)

    # Merge on comp_id and atom_id_2
    merged_2 = result_df.merge(
        bond_df,
        left_on=['comp_id', join_column],
        right_on=['comp_id', 'atom_id_2'],
        suffixes=('', f'_bond{join_number}')
    )

    # Rename atom_id_2 to atom_id_joined and drop atom_id_1
    merged_2.rename(columns={'atom_id_1': f'atom_id_joined{join_number}'}, inplace=True)
    merged_2.drop(columns=['atom_id_2'], inplace=True)

    # Concatenate the two merged DataFrames
    result_df = pd.concat([merged_1, merged_2], ignore_index=True)
    #print(result_df[(result_df['comp_id']=='001') & (result_df['atom_id']=='C02')])

    # Drop rows where atom_id is the same as f'atom_id_joined{join_number}'
    result_df = result_df[result_df['atom_id'] != result_df[f'atom_id_joined{join_number}']]

    # Drop duplicate rows based on atom_id and f'atom_id_joined{join_number}'
    result_df = result_df.drop_duplicates(subset=['comp_id', 'atom_id', f'atom_id_joined{join_number}'])

    return result_df


def modify_columns(atom_df, bond_df):
        # make new column atomic number and depending on the type symbol used, give the atomic number
        # E.g. in the periodic table, the atom C has the atomic number 6
        atom_df['atomic_number'] = atom_df['type_symbol'].apply(lambda x: gemmi.Element(x).atomic_number)

        # Define the mapping from string to integer
        value_order_mapping = {
            'SING': 1,
            'DOUB': 2,
            'TRIP': 3
        }

        # Replace the values in the 'value_order' column
        bond_df['value_order'] = bond_df['value_order'].replace(value_order_mapping)

        return atom_df, bond_df


def check_assertions(atom_df, bond_df): # CAN BE EXTENDED: more checks
    # Before cleaning nan and ? rows, check that we did not impute these values by ourselves
    assert atom_df['is_hydrogen'].notna().all(), f"NaN values found in column 'is_hydrogen'"
    assert atom_df['bonded_hydrogens'].notna().all(), f"NaN values found in column 'bonded_hydrogens'"


def filter_centralatom_bondedhydrogens(atom_df, central_atom=None, num_hydrogens=None):
        if central_atom is None and num_hydrogens is None:
            print("Please specify one of the filtering parameters: central_atom or num_hydrogens")
            print("atom_df is returned without filtering")
            return atom_df
        elif central_atom is None:
            atom_df_filtered = atom_df.loc[atom_df['bonded_hydrogens'] == Config.num_hydrogens]
        elif num_hydrogens is None:
            atom_df_filtered = atom_df.loc[atom_df['type_symbol'] == Config.central_atom]
        else:
            atom_df_filtered = atom_df.loc[(atom_df['type_symbol'] == Config.central_atom) & (atom_df['bonded_hydrogens'] == Config.num_hydrogens)]
        print("Size of Filtered dataframe: ", atom_df_filtered.shape[0])
        return atom_df_filtered


def filter_num_neighbors_to_centralatom(df, num_neighbors_to_centralatom):
    #result_df_new = result_df.groupby(['comp_id', 'atom_id']).size().reset_index(name='counts')
    grouped = df.groupby(['comp_id', 'atom_id'])
    filtered_groups = grouped.filter(lambda x: len(x) == num_neighbors_to_centralatom)
    return filtered_groups


def filter_allowed_atomic_numbers(df, allowed_atomic_numbers):
    # Step 1: Find all columns that begin with 'atomic_number'
    atomic_columns = [col for col in df.columns if col.startswith('atomic_number')]
    print(atomic_columns)

    # Step 2: Check for each row whether all atomic_number columns have values in valid_values
    invalid_rows_mask = df[atomic_columns].apply(lambda row: not all(val in allowed_atomic_numbers for val in row if pd.notna(val)), axis=1)

    # Step 3: Identify rows that do not fulfill the valid values condition
    invalid_rows = df[invalid_rows_mask]

    # Step 4: Extract comp_id and atom_id values of these rows
    invalid_comp_atom_ids = invalid_rows[['comp_id', 'atom_id']]

    # Step 5: Delete all rows that have the same comp_id and atom_id as any of the identified rows
    filtered_df = df[~df[['comp_id', 'atom_id']].apply(tuple, axis=1).isin(invalid_comp_atom_ids.apply(tuple, axis=1))]
    
    return filtered_df


# Function to create feature vectors
def create_feature_vectors_relative(df):
    feature_vectors_dict = {}

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        comp_id = row['comp_id']
        atom_id = row['atom_id']
        key = (comp_id, atom_id)
        
        if key not in feature_vectors_dict:
            feature_vectors_dict[key] = [
                row['atomic_number']
            ]

        # Append relevant columns
        feature_vectors_dict[key].extend([
            row['atomic_number_coor_joined1'],
            row['value_order'],
            float(row['model_Cartn_x_coor_joined1']) - float(row['model_Cartn_x']),
            float(row['model_Cartn_y_coor_joined1']) - float(row['model_Cartn_y']),
            float(row['model_Cartn_z_coor_joined1']) - float(row['model_Cartn_z'])
        ])

    # Convert the dictionary to a list of feature vectors
    feature_vectors = [features for features in feature_vectors_dict.values()]
    
    return feature_vectors


def separate_hydrogens(feature_vectors):
    X, y = [], []
    for vec in tqdm(feature_vectors):
        assert vec[0] != 1, "Central atom is hydrogen atom, Aborted program"
        X_auxiliary = []
        X_auxiliary.append(vec[0])
        found_hydrogen = False
        for i in range(1, len(vec), 5):
            if vec[i] == 1: # hydrogen case
                assert vec[i+1] == 1, "Hydrogen atom does not have one bonding as it should have"
                assert not found_hydrogen, "Found more than one bonding to hydrogen"
                y.append(vec[i+2:i+5])
                found_hydrogen = True
            else:
                X_auxiliary.extend(vec[i:i+5])
        X.append(X_auxiliary)
    return X, y


def zero_padding(X):
    max_length = 0
    for vec in X:
        if len(vec) > max_length:
            max_length = len(vec)
    print("Max feature vector length is: ", max_length)

    # Pad the lists with zeros to the maximum length
    X_padded = [lst + [0] * (max_length - len(lst)) for lst in X]

    # Now X_padded can be converted to an numpy array because of homogenous shape
    X_padded = np.asarray(X_padded)

    print(X_padded.shape)
    return X_padded, max_length


def append_descriptor(X_padded, max_length):
    cm = CoulombMatrix(n_atoms_max=int((max_length-1)/5) + 1, permutation='none')
    X_padded_descriptor = []
    for vec in tqdm(X_padded):
        X_auxiliary = []
        X_auxiliary.extend(vec)

        # collect parameters for the descriptor
        symbols = [gemmi.Element(int(vec[0])).name]
        positions = [np.array([0, 0, 0])]
        for i in range(1, len(vec), 5):
            if vec[i] != 0: # check that we are not in the zero padding
                symbols.append(gemmi.Element(int(vec[i])).name)
                positions.append(vec[i+2:i+5])

        # create descriptor
        atoms_object = Atoms(symbols=symbols, positions=positions)
        atoms_object_descriptor = cm.create(atoms_object)

        # append to the current zero padded feature vector
        X_auxiliary.extend(atoms_object_descriptor)
        X_padded_descriptor.append(X_auxiliary)
    return X_padded_descriptor


def one_hot_encoding(X_padded):
    # One-hot encoding functions
    def one_hot_encode_centralatom(value):
        if value == 0:  # for the zero padded values
            return [0, 0, 0, 0]
        elif value == 6:
            return [1, 0, 0, 0]
        elif value == 7:
            return [0, 1, 0, 0]
        elif value == 8:
            return [0, 0, 1, 0]
        elif value == 16:
            return [0, 0, 0, 1]
        else:
            raise Exception("Unwanted atom type!")

    def one_hot_encode_valueorder(value):
        if value == 0:  # for the zero padded values
            return [0, 0, 0]
        elif value == 1:
            return [1, 0, 0]
        elif value == 2:
            return [0, 1, 0]
        elif value == 3:
            return [0, 0, 1]
        else:
            raise Exception("Unwanted value order!")

    # determine central atom indices to encode
    centralatom_indices_to_encode = list(range(1, max_length, 5))
    print("Central atom indices: ", centralatom_indices_to_encode)

    # determine value order indices to encode
    valueorder_indices_to_encode = list(range(2, max_length, 5))
    print("Value order indices: ", valueorder_indices_to_encode)

    # Create the new array with one-hot encoded values
    X_encoded = []
    for sample in tqdm(X_padded):
        new_sample = []
        for i, value in enumerate(sample):  # max_length 
            if i in centralatom_indices_to_encode:
                new_sample.extend(one_hot_encode_centralatom(value))
            elif i in valueorder_indices_to_encode:
                new_sample.extend(one_hot_encode_valueorder(value))
            else:
                new_sample.append(value)
        X_encoded.append(new_sample)
    return X_encoded


def make_training_and_testing_set(X, y):
    # Create the directory if it doesn't exist
    directory = os.path.dirname(Config.training_set_save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Save the training data
    with open(Config.training_set_save_path, 'wb') as f:
        pickle.dump((X_train, y_train), f)

    # Save the testing data
    with open(Config.testing_set_save_path, 'wb') as f:
        pickle.dump((X_test, y_test), f)


if __name__ == "__main__":
    print("BEGIN")
    if not Config.dev:
        doc = cif.read_file(Config.cif_path)
        print('Number of blocks/molecules in dataset: '+ str(len(doc)))

        if Config.make_bond_df:
            print("Making bond_df")
            create_bond_df(doc)
            print("Finished making bond_df")
        
        if Config.make_atom_df:
            print("Making atom_df")
            create_atom_df(doc)
            print("Finished making atom_df")

        if Config.make_atom_df_extended:
            try:
                atom_df = pd.read_csv(Config.atom_df_path)
                bond_df = pd.read_csv(Config.bond_df_path)
            except FileNotFoundError as e:
                print(f"File not found: {e.filename}")
            except pd.errors.EmptyDataError as e:
                print(f"No data: {e}")
            except pd.errors.ParserError as e:
                print(f"Parsing error: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

            print("Making atom_df_extended")
            create_atom_df_extended(atom_df, bond_df)
            print("Finished making atom_df_extended")

        # load atom_df and bond_df
        atom_df = pd.read_csv(Config.atom_df_filtered_path)
        bond_df = pd.read_csv(Config.bond_df_path)

        # check certain assertions
        print("Making assertions about dataframes")
        check_assertions(atom_df, bond_df)

        # clean dataframe (remove nan and ? samples)
        print("Cleaning dataframes")
        atom_df, bond_df = clean_dataframe(atom_df, bond_df)

        # Modify columns (add atomic number to atom df and change value order from string to integer)
        print("modifying columns of dataframes")
        atom_df, bond_df = modify_columns(atom_df, bond_df)

        # Save the updated atom_df to CSV
        atom_df.to_csv('../data/atom_df_extended_filtered.csv', index=False)
        bond_df.to_csv('../data/bond_df_filtered.csv', index=False)
    else:
        # load atom_df and bond_df
        atom_df = pd.read_csv('../data/atom_df_extended_filtered.csv')
        bond_df = pd.read_csv('../data/bond_df_filtered.csv')

        # filtering
        atom_df_filtered = filter_centralatom_bondedhydrogens(atom_df, Config.central_atom, Config.num_hydrogens)

        # remove unneeded columns
        atom_df_filtered.drop(columns=["type_symbol", "is_hydrogen", "bonded_hydrogens"], inplace=True)
        atom_df.drop(columns=["type_symbol", "is_hydrogen", "bonded_hydrogens"], inplace=True)

        # make the big table for easier feature vector generation
        result_df_old = atom_df_filtered
        df_per_depth = []
        for depth in range(1, Config.max_neighbor_depth+1):
            # Join with bonds
            result_df_new = join_bonds(result_df_old, bond_df, join_number=depth)

            # Join with atoms
            result_df_final = join_coordinates(result_df_new, atom_df, join_number=depth)

            # Determine the sorting values if more than 1 neighbor depth is used
            sorting_values = ['comp_id', 'atom_id']
            if depth >= 1:
                for i in range(1, depth+1):
                    sorting_values.append(f'atom_id_joined{i}')

            # Sort the result_df
            result_df_final_sorted = result_df_final.sort_values(by=sorting_values, inplace=False).reset_index(drop=True)

            # Add to df_per_depth list
            df_per_depth.append(result_df_final_sorted.copy())

            # Set new to old
            result_df_old = result_df_final_sorted

        # initialize preventive
        result_df_filtered = result_df_final_sorted

        # filter out samples where atoms are not of atom type 'H', 'C', 'O', 'N', 'S'
        if Config.allowed_atomic_numbers is not None:
            before = result_df_filtered.shape[0]
            result_df_filtered = filter_allowed_atomic_numbers(result_df_final_sorted, Config.allowed_atomic_numbers)
            diff = before - result_df_filtered.shape[0] 
            print(f"Filtering out samples with unwanted atom types removed {diff} many rows")
        
        # filter out samples where the central atom has more than 4 neighbors (including H atom)
        if Config.num_neighbors_to_centralatom is not None:
            before = result_df_filtered.shape[0]
            result_df_filtered = filter_num_neighbors_to_centralatom(result_df_filtered, Config.num_neighbors_to_centralatom)
            diff = before - result_df_filtered.shape[0] 
            print(f"Filtering out samples with incorrect number of neighbors removed {diff} many rows")

        ### Training / Testing dataset generation

        # sort after comp_id, atom_id, atomic_number_coor_joined1, value_order, ...
        sorting_values = ['comp_id', 'atom_id', 'atomic_number_coor_joined1', 'value_order']
        result_df_filtered = result_df_filtered.sort_values(by=sorting_values, inplace=False).reset_index(drop=True)

        # create feature vectors (through sorting before, this gets quite permutation-invariant)
        print("Computing feature vectors")
        feature_vectors = create_feature_vectors_relative(result_df_filtered)

        # separate hydrogens from the feature vector (only works for one missing hydrogen)
        print("Separating hydrogens from feature vectors")
        X, y = separate_hydrogens(feature_vectors)

        # apply zero padding
        print("Applying zero padding")
        X, max_length = zero_padding(X)

        # apply descriptor (can be left out)
        print("Applying the descriptor")
        X = append_descriptor(X, max_length)

        # apply one hot encoding
        print("Applying one hot encoding")
        X = one_hot_encoding(X)

        # make training and testing dataset
        print("Making the training and testing dataset")
        make_training_and_testing_set(X, y)

        print("FINISH")


# """
# working method do recursively find all neighbors of a given current atom (e.g. the central atom)
# problem: method is very slow and therefore not applicable to large datasets 
# """
# def recursive_neighbor_finder(atom_df, bond_df, current_atom, neighbor_list, current_depth, current_value_order=-1, max_depth=1):
#     if current_depth > max_depth:
#         return True  # Continue processing

#     comp_id, atom_id = current_atom

#     # Find the current atom in atom_df and save its coordinates along with comp_id and atom_id
#     atom_info = atom_df[(atom_df['comp_id'] == comp_id) & (atom_df['atom_id'] == atom_id)]
#     assert len(atom_info) == 1, "atom not found"

#     # If atom occurrs that is unwanted, stop the computation process
#     current_atomic_number = atom_info.iloc[0]['atomic_number']
#     if current_atomic_number not in Config.allowed_atomic_numbers:
#         return False  # Abort the whole computation

#     # Add infos of the current atom to the neighbor list
#     if not atom_info.empty:
#         x, y, z = atom_info.iloc[0]['model_Cartn_x'], atom_info.iloc[0]['model_Cartn_y'], atom_info.iloc[0]['model_Cartn_z']
#         if current_depth not in neighbor_list:
#             neighbor_list[current_depth] = []
#         neighbor_list[current_depth].append((comp_id, atom_id, current_atomic_number, current_value_order, x, y, z))

#     # Find all bonds in bond_df involving the current atom
#     bonds_involving_atom = bond_df[(bond_df['comp_id'] == comp_id) & 
#                                    ((bond_df['atom_id_1'] == atom_id) | (bond_df['atom_id_2'] == atom_id))]

#     # For each bond, determine the neighboring atom and recursively call the method
#     for _, bond in bonds_involving_atom.iterrows():
#         bond_value_order = bond['value_order']
#         if bond['atom_id_1'] == atom_id:
#             neighbor_atom = (bond['comp_id'], bond['atom_id_2'])
#         else:
#             neighbor_atom = (bond['comp_id'], bond['atom_id_1'])

#         # Check if the neighbor_atom has already been handled
#         already_handled = any(neighbor_atom == (comp_id, atom_id) for depth, neighbors in neighbor_list.items() for comp_id, atom_id, _, _, _, _, _ in neighbors)
#         if not already_handled:
#             # Recursively call the function for the neighboring atom
#             if not recursive_neighbor_finder(atom_df, bond_df, neighbor_atom, neighbor_list, current_depth + 1, bond_value_order, max_depth):
#                 return False # Abort if an unwanted type is found
    
#     return True # Continue processing

        
        # # Example:
        # neighbor_list = {}
        # current_atom = ('001', 'C06')  # Example atom
        # recursive_neighbor_finder(atom_df, bond_df, current_atom, neighbor_list, current_depth=0, current_value_order=-1, max_depth=Config.max_neighbor_depth)
        # print(neighbor_list)