from gemmi import cif
import gemmi
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import Config
from sklearn.model_selection import train_test_split
import pickle
import os
from collections import deque
#import sys


def create_bond_df(doc):
    bond_site_cols = ['comp_id', 'atom_id_1', 'atom_id_2', 'value_order']

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


def modify_columns(atom_df, bond_df):
        # make new column atomic number and depending on the type symbol used, give the atomic number
        # E.g. in the periodic table, the atom C has the atomic number 6
        atom_df['atomic_number'] = atom_df['type_symbol'].apply(lambda x: gemmi.Element(x).atomic_number)

        # Replace the values in the 'value_order' column
        bond_df['value_order'] = bond_df['value_order'].replace(Config.value_order_mapping)

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
        print("Size of Filtered atom_df dataframe: ", atom_df_filtered.shape[0])
        return atom_df_filtered


def create_adjacency_matrix_np_vectorized(comp_id, bond_df):
    # Filter rows for the given comp_id
    comp_df = bond_df[bond_df['comp_id'] == comp_id]

    # If no bonds are found, return empty structures
    if comp_df.empty:
        return np.array([]), [], {}
    
    # Get unique atoms in the molecule
    atoms = pd.unique(comp_df[['atom_id_1', 'atom_id_2']].values.ravel('K'))
    atom_index = {atom: idx for idx, atom in enumerate(atoms)}
    
    # Initialize adjacency matrix with zeros, ensuring correct dtype
    atom_count = len(atoms)
    adjacency_matrix = np.zeros((atom_count, atom_count), dtype=np.int32)
    
    # Map atom ids and value_order to indices and numeric values
    indices_1 = comp_df['atom_id_1'].map(atom_index).values
    indices_2 = comp_df['atom_id_2'].map(atom_index).values
    values = comp_df['value_order'].values
    
    # Use numpy advanced indexing to set the adjacency matrix
    adjacency_matrix[indices_1, indices_2] = values
    adjacency_matrix[indices_2, indices_1] = values  # assuming undirected graph
    
    return adjacency_matrix, atoms, atom_index


def depth_limited_search_with_features(start_atom, adj_matrix, atom_index, atom_info_dict, depth=1):
    if start_atom not in atom_index:
        return {}

    start_index = atom_index[start_atom]
    start_atom_info = atom_info_dict[start_atom]
    visited = set()
    queue = deque([(start_index, 0, 1)])  # Queue has tuples (index for adj. matrix, list index for features_by_depth[current_depth], current depth); Start with depth 1
    features_by_depth = {}
    connections = {}

    while queue:
        current_index, list_index, current_depth = queue.popleft()
        if current_depth > depth:
            continue
        if current_index not in visited:
            visited.add(current_index)
            if current_depth not in features_by_depth:
                features_by_depth[current_depth] = []
                connections[current_depth] = {}

            connections[current_depth][list_index] = []

            # Gather all neighbors
            neighbors = []
            for neighbor_index, bond_order in enumerate(adj_matrix[current_index]):
                if bond_order > 0 and neighbor_index not in visited:
                    neighbor_atom = list(atom_index.keys())[list(atom_index.values()).index(neighbor_index)]
                    atom_info = atom_info_dict.get(neighbor_atom)
                    if atom_info is None or atom_info[0] is None:
                        # Encountered an atom with an unspecified atomic number, return an empty list
                        return {}, {}
                    neighbors.append((neighbor_index, neighbor_atom, bond_order))

            # Sort neighbors first by atomic number, then by bond order
            neighbors.sort(key=lambda x: (atom_info_dict[x[1]][0], x[2]))

            # Loop through each neighbor of the current atom
            for neighbor_index, neighbor_atom, bond_order in neighbors:
                atom_info = atom_info_dict.get(neighbor_atom)
                # Compute relative coordinates
                rel_x = atom_info[1] - start_atom_info[1]
                rel_y = atom_info[2] - start_atom_info[2]
                rel_z = atom_info[3] - start_atom_info[3]

                # Add new feature values
                feature_values = [atom_info[0], bond_order, rel_x, rel_y, rel_z]
                next_list_index = len(features_by_depth[current_depth]) # needs to be computed before adding values to the the correct list index
                features_by_depth[current_depth].extend(feature_values)

                # Update the connections and the queue
                connections[current_depth][list_index].append(next_list_index)
                queue.append((neighbor_index, next_list_index, current_depth + 1))

    return features_by_depth, connections


def preprocess_data(bond_df, atom_df, comp_ids):
    adj_matrices = {}
    atom_indices = {}
    atom_infos = {}

    for comp_id in tqdm(comp_ids, position=0, leave=True):
        adj_matrix, atoms, atom_index = create_adjacency_matrix_np_vectorized(comp_id, bond_df)
        adj_matrices[comp_id] = adj_matrix
        atom_indices[comp_id] = atom_index

        # Precompute atom information for fast lookup
        atom_info_dict = {}
        atom_comp_df = atom_df[atom_df['comp_id'] == comp_id]
        for _, row in atom_comp_df.iterrows():
            atom_info_dict[row['atom_id']] = (
                Config.atomic_number_mapping.get(row['type_symbol']),
                float(row['model_Cartn_x']),
                float(row['model_Cartn_y']),
                float(row['model_Cartn_z'])
            )
        atom_infos[comp_id] = atom_info_dict
    
    return adj_matrices, atom_indices, atom_infos


def convert_connections_to_tuples(connections):
    connection_tuples = {}

    for atom_id, connections_by_depth in connections.items():
        offset = - 5  # because hydrogen atom will not be added to connections
        connection_tuples[atom_id] = []
        for depth, indices in connections_by_depth.items():
            #print(f"At depth {depth} we have offset {offset}")
            max_index = 0
            for parent_index, child_indices in indices.items():
                for child_index in child_indices:
                    if child_index > max_index:
                        max_index = child_index
                    if depth == 1:
                        if child_index != 0:  # do not add the connection to hydrogen atom because it is removed later anyway
                            connection_tuples[atom_id].append((-1, child_index + offset))
                    else:
                        connection_tuples[atom_id].append((parent_index + old_offset, child_index + offset))
            old_offset = offset  # Keep the old offset for adding to the parent index
            offset += max_index + 5  # Update the offset for the next depth level
    
    return connection_tuples


def filter_num_neighbors_to_centralatom(results, num_neighbors):
    delete_keys = []
    for (comp_id, atom_id), depths in results.items():
        if depths:
            # Save comp IDs and atom IDs with not num_neighbors neighbors at depth 1
            if 1 in depths and len(depths[1]) != num_neighbors * 5:
                delete_keys.append((comp_id, atom_id))
    # Remove the keys
    for (comp_id, atom_id) in delete_keys:
        del results[(comp_id, atom_id)]


def restructure_connections(connections):
    restructured_connections = {}
    for comp_id, _ in connections.items():
        restructured = convert_connections_to_tuples(connections[comp_id])
        for atom_id, new_connections in restructured.items():
            restructured_connections[(comp_id, atom_id)] = new_connections
    return restructured_connections


def restructure_results(results):
    restructured_results = {}
    for comp_id, atoms in results.items():
        for atom_id, depths in atoms.items():
            restructured_results[(comp_id, atom_id)] = depths
    return restructured_results


def filtered_results(results, connections, num_neighbors):
    delete_keys = []
    for (comp_id, atom_id), depths in results.items():
        if depths:
            # Save comp IDs and atom IDs with not num_neighbors neighbors at depth 1
            if 1 in depths and len(depths[1]) != num_neighbors * 5:
                delete_keys.append((comp_id, atom_id))
    # Remove the keys
    for (comp_id, atom_id) in delete_keys:
        del results[(comp_id, atom_id)]
        del connections[(comp_id, atom_id)]


def seperate_hydrogens(results):
    y = {}
    for (comp_id, atom_id), depths in results.items():
        if depths:
            # Remove hydrogen atoms and collect their coordinates in y
            if 1 in depths:
                depth_list = depths[1]
                if len(depth_list) % 5 != 0:
                    raise Exception("depth list is not consistently 5 tuples")
                    continue
                i = 0
                y[(comp_id, atom_id)] = []
                while i < len(depth_list):
                    if depth_list[i] == 1:  # Hydrogen atom
                        y[(comp_id, atom_id)].append([depth_list[i + 2], depth_list[i + 3], depth_list[i + 4]])
                        del depth_list[i:i + 5]
                    else:
                        i += 5
    return results, y


def one_hot_encoding(X_padded):
    # One-hot encoding functions
    def one_hot_encode_centralatom(value):
        if value == 0:  # for the zero padded values
            return [0, 0, 0, 0, 0]
        elif value == 1:
            return [1, 0, 0, 0, 0]
        elif value == 6:
            return [0, 1, 0, 0, 0]
        elif value == 7:
            return [0, 0, 1, 0, 0]
        elif value == 8:
            return [0, 0, 0, 1, 0]
        elif value == 16:
            return [0, 0, 0, 0, 1]
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

    encoded_features = []
    for i in range(0, len(X_padded), 5):
        atom_type = X_padded[i]
        bond_order = X_padded[i + 1]
        relative_coords = X_padded[i + 2: i + 5]
        encoded_atom_type = one_hot_encode_centralatom(atom_type)
        encoded_bond_order = one_hot_encode_valueorder(bond_order)
        encoded_features.extend(encoded_atom_type + encoded_bond_order + relative_coords)

    return encoded_features


def process_results(results, depth):
    processed_features = {}
    max_lengths = {d: 0 for d in range(1, depth + 1)}

    # First pass to determine the maximum length for zero padding
    for (comp_id, atom_id), depths in results.items():
        for d in range(1, depth + 1):
            if d in depths and len(depths[d]) > max_lengths[d]:
                max_lengths[d] = len(depths[d])

    for (comp_id, atom_id), depths in results.items():
        # Zero padding
        for d in range(1, depth + 1):
            if d in depths:
                while len(depths[d]) < max_lengths[d]:
                    depths[d].extend([0, 0, 0, 0, 0])

        # Merge lists
        merged_list = []
        for d in range(1, depth + 1):
            if d in depths:
                merged_list.extend(depths[d])

        # One-hot encoding
        encoded_features = one_hot_encoding(merged_list)
        processed_features[(comp_id, atom_id)] = encoded_features

    return processed_features


def save_dataset(X, y, connections):  # inputs are all dicts
    # Create the directory if it doesn't exist
    directory = os.path.dirname(Config.dataset_save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the training data
    with open(Config.dataset_save_path, 'wb') as f:
        pickle.dump((X, y, connections), f)


def make_training_and_testing_set(X, y, connections):  # inputs are all lists
    # Create the directory if it doesn't exist
    directory = os.path.dirname(Config.training_set_save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Split the data
    X_train, X_test, y_train, y_test, con_train, con_test = train_test_split(X, y, connections, test_size=Config.train_test_split_ratio, random_state=Config.random_state)

    # Saving the numpy arrays into a single .npz file
    np.savez(Config.training_set_save_path, X_train=X_train, y_train=y_train)
    np.savez(Config.testing_set_save_path, X_test=X_test, y_test=y_test)

    # Save connections with pickle (np.savez not possible because connections has no homogenous shape)
    with open(Config.connections_set_save_path, 'wb') as f:
        pickle.dump((con_train, con_test), f)


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

        # load atom_df and bond_df
        atom_df = pd.read_csv('../data/atom_df_extended_filtered.csv')
        bond_df = pd.read_csv('../data/bond_df_filtered.csv')

        # filtering
        atom_df_filtered = filter_centralatom_bondedhydrogens(atom_df, Config.central_atom, Config.num_hydrogens)

        # remove unneeded columns
        atom_df_filtered.drop(columns=["type_symbol", "is_hydrogen", "bonded_hydrogens"], inplace=True)
        atom_df.drop(columns=["is_hydrogen", "bonded_hydrogens"], inplace=True)  # NOT! atom_df.drop(columns=["type_symbol", "is_hydrogen", "bonded_hydrogens"], inplace=True)

        # Preprocess data (i.e. make adjacency matrices for each comp_id to make the further computations faster afterwards)
        if Config.make_preprocessed_data:
            # Preprocess data
            print("Making preprocessed data")
            comp_ids = atom_df['comp_id'].unique() 
            print("Number of comp_ids: ", len(comp_ids))
            adj_matrices, atom_indices, atom_infos = preprocess_data(bond_df, atom_df, comp_ids)

            # Save to pickle file
            with open(Config.preprocessed_data_save_path, 'wb') as f:
                pickle.dump((adj_matrices, atom_indices, atom_infos), f)

        # Open and load the pickle file
        with open(Config.preprocessed_data_save_path, 'rb') as f:
            preprocessed_data = pickle.load(f)
        adj_matrices, atom_indices, atom_infos = preprocessed_data

        # Process only comp_ids in atom_df_filtered
        print("Making Depth First Search for Neighbor Finding + Feature Computation")
        filtered_comp_ids = atom_df_filtered['comp_id'].unique()
        results = {}
        all_connections = {}

        for comp_id in tqdm(filtered_comp_ids, position=0, leave=True):
            adj_matrix = adj_matrices[comp_id]
            atom_index = atom_indices[comp_id]
            atom_info_dict = atom_infos[comp_id]
            features_by_depth = {}
            connections_by_depth = {}
            
            # Process only atoms in atom_df_filtered for the current comp_id
            filtered_atoms = atom_df_filtered[atom_df_filtered['comp_id'] == comp_id]['atom_id'].unique()
            for atom in filtered_atoms:
                features, connections = depth_limited_search_with_features(atom, adj_matrix, atom_index, atom_info_dict, depth=Config.neighbor_depth)
                if features:
                    features_by_depth[atom] = features
                    connections_by_depth[atom] = connections
            if features_by_depth:
                results[comp_id] = features_by_depth
                all_connections[comp_id] = connections_by_depth

        # Save to pickle file (for debugging if wanted)
        # with open(f'../data/{Config.base_folder_name}/intermediate-results.pkl', 'wb') as f:
        #     pickle.dump((results, all_connections), f)

        # Display the results for one comp_id
        for comp_id, features in results.items():
            print(f"Features by depth for comp_id {comp_id}:")
            for start_atom, depth_features in features.items():
                print(f"  Start atom {start_atom}: {depth_features}")
            print()
            break

        # make a one-level dictionary with tuple keys from a two level dictionary
        results = restructure_results(results)
        all_connections = restructure_connections(all_connections)

        # filter out all samples that do not have exactly Config.num_neighbors_to_centralatom neighbors to the central atom
        print(f"Filtering out samples that do not have {Config.num_neighbors_to_centralatom} neighbors to the central atom")
        filtered_results(results, all_connections, num_neighbors=Config.num_neighbors_to_centralatom)

        # seperate the hydrogens from the neighbors of depth 1 (as we want to predict those positions) and put them to y
        print("Separating hydrogens")
        results, y = seperate_hydrogens(results) 

        # There should be as many keys in results as in y
        assert len(results) == len(y) and len(results) == len(all_connections)

        # make zero padding, merge feature lists of different depths and apply one-hot encoding
        print("Making the final feature vector X")
        X = process_results(results, Config.neighbor_depth)

        # There should be as many keys in X as in y
        assert len(X) == len(y) and len(X) == len(all_connections)

        # Print the first 5 keys directly
        print("5 example keys of the dataset: ")
        for i, key in enumerate(X.keys()):
            if i < 5:
                print(key, end=' ')
            else:
                break

        # Save the dataset, where X and y are dictionaries at the moment (with the IDs)
        save_dataset(X, y, all_connections)

        # Make X, y and all_connections to lists (they were dictionaries before)
        X_raw = []
        y_raw = []
        connections_raw = []
        for (comp_id, atom_id), feature_list in X.items():
            X_raw.append(feature_list)
            y_raw.extend(y[(comp_id, atom_id)])
            connections_raw.append(all_connections[(comp_id, atom_id)])
        
        assert len(X_raw) == len(y_raw) and len(X_raw) == len(connections_raw)
        print(f"There are {len(X_raw)} samples in the dataset")

        # Split the data and save it
        make_training_and_testing_set(X_raw, y_raw, connections_raw)

        # Save the configuration
        Config.save_to_json(Config.config_save_path)

        print("FINISH")
