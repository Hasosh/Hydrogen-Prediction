from gemmi import cif
import gemmi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import Config


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
    bond_df.to_csv('../data/bond_df.csv', index=False)

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
    atom_df.to_csv('../data/atom_df.csv', index=False)

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
    atom_df.to_csv('../data/atom_df_extended.csv', index=False)

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
    total_comp_ids_to_delete = nan_comp_ids_to_delete.union(question_mark_comp_ids_to_delete)

    # Final row counts
    final_atom_row_count = len(atom_df)
    final_bond_row_count = len(bond_df)

    # Print results
    print(f"Identified {len(nan_comp_ids_to_delete)} unique comp_ids with NaN values.")
    print(f"Identified {len(question_mark_comp_ids_to_delete)} unique comp_ids with '?' values in atom_df.")
    print(f"Deleted {initial_atom_row_count - final_atom_row_count} rows from atom_df.")
    print(f"Deleted {initial_bond_row_count - final_bond_row_count} rows from bond_df.")
    
    return atom_df, bond_df


# Function to determine the correct join column and perform the join
def join_atom_coordinates(result_df, atom_df):
    # Create a copy of the DataFrame to avoid modifying the original one
    df = result_df.copy()

    # Determine the correct join column
    df['join_on_atom_id'] = df.apply(
        lambda row: row['atom_id_2'] if row['atom_id'] == row['atom_id_1'] else row['atom_id_1'], axis=1
    )

    # Perform the join
    merged_df = df.merge(
        atom_df[['comp_id', 'atom_id', 'atomic_number', 'model_Cartn_x', 'model_Cartn_y', 'model_Cartn_z']],
        left_on=['comp_id', 'join_on_atom_id'],
        right_on=['comp_id', 'atom_id'],
        suffixes=('', '_joined')
    )

    # Extract and rename the coordinates columns
    merged_df = merged_df.rename(
        columns={
            'model_Cartn_x_joined': 'model_Cartn_x2',
            'model_Cartn_y_joined': 'model_Cartn_y2',
            'model_Cartn_z_joined': 'model_Cartn_z2'
        }
    )

    # Keep only the necessary columns
    result_df = merged_df[
        ['comp_id', 'atom_id', 'type_symbol', 'model_Cartn_x', 'model_Cartn_y', 'model_Cartn_z', 'is_hydrogen', 
         'bonded_hydrogens', 'atomic_number', 'atom_id_1', 'atom_id_2', 'value_order', 'model_Cartn_x2', 'model_Cartn_y2', 'model_Cartn_z2', 'atomic_number_joined']
    ]

    return result_df


def recursive_neighbor_finder(atom_df, bond_df, current_atom, neighbor_list, current_depth, current_value_order=-1, max_depth=1):
    if current_depth > max_depth:
        return

    comp_id, atom_id = current_atom

    # Find the current atom in atom_df and save its coordinates along with comp_id and atom_id
    atom_info = atom_df[(atom_df['comp_id'] == comp_id) & (atom_df['atom_id'] == atom_id)]
    assert len(atom_info) == 1, "atom not found"

    # Add infos of the current atom to the neighbor list
    if not atom_info.empty:
        current_atomic_number = atom_info.iloc[0]['atomic_number']
        x, y, z = atom_info.iloc[0]['model_Cartn_x'], atom_info.iloc[0]['model_Cartn_y'], atom_info.iloc[0]['model_Cartn_z']
        if current_depth not in neighbor_list:
            neighbor_list[current_depth] = []
        neighbor_list[current_depth].append((comp_id, atom_id, current_atomic_number, current_value_order, x, y, z))

    # Find all bonds in bond_df involving the current atom
    bonds_involving_atom = bond_df[(bond_df['comp_id'] == comp_id) & 
                                   ((bond_df['atom_id_1'] == atom_id) | (bond_df['atom_id_2'] == atom_id))]

    # For each bond, determine the neighboring atom and recursively call the method
    for _, bond in bonds_involving_atom.iterrows():
        bond_value_order = bond['value_order']
        if bond['atom_id_1'] == atom_id:
            neighbor_atom = (bond['comp_id'], bond['atom_id_2'])
        else:
            neighbor_atom = (bond['comp_id'], bond['atom_id_1'])

        # Check if the neighbor_atom has already been handled
        already_handled = any(neighbor_atom == (comp_id, atom_id) for depth, neighbors in neighbor_list.items() for comp_id, atom_id, _, _, _, _, _ in neighbors)
        if not already_handled:
            # Recursively call the function for the neighboring atom
            recursive_neighbor_finder(atom_df, bond_df, neighbor_atom, neighbor_list, current_depth + 1, bond_value_order, max_depth)




if __name__ == "__main__":
    if not Config.dev:
        doc = cif.read_file("../data/components.cif")
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
                atom_df = pd.read_csv('../data/atom_df.csv')
                bond_df = pd.read_csv('../data/bond_df.csv')
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
        atom_df = pd.read_csv('../data/atom_df_extended.csv')
        bond_df = pd.read_csv('../data/bond_df.csv')

        # Before cleaning nan and ? rows, check that we did not impute these values by ourselves
        assert atom_df['is_hydrogen'].notna().all(), f"NaN values found in column 'is_hydrogen'"
        assert atom_df['bonded_hydrogens'].notna().all(), f"NaN values found in column 'bonded_hydrogens'"

        # clean dataframe
        atom_df, bond_df = clean_dataframe(atom_df, bond_df)

        ### Modify columns
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

        # Save the updated atom_df to CSV
        atom_df.to_csv('../data/atom_df_extended_filtered.csv', index=False)
        bond_df.to_csv('../data/bond_df_filtered.csv', index=False)
    else:
        # load atom_df and bond_df
        atom_df = pd.read_csv('../data/atom_df_extended_filtered.csv')
        bond_df = pd.read_csv('../data/bond_df_filtered.csv')

        ### Training / Testing dataset generation
        atom_df_filtered = atom_df.loc[(atom_df['type_symbol'] == Config.central_atom) & (atom_df['bonded_hydrogens'] == Config.num_hydrogens)]

        neighbor_list = {}
        current_atom = ('001', 'C02')  # Example atom
        recursive_neighbor_finder(atom_df, bond_df, current_atom, neighbor_list, current_depth=0, current_value_order=-1, max_depth=Config.max_neighbor_depth)
        print(neighbor_list)

        # # Merge on comp_id and atom_id_1
        # merged_1 = atom_df_filtered.merge(bond_df, left_on=['comp_id', 'atom_id'], right_on=['comp_id', 'atom_id_1'], suffixes=('', '_bond_1'))

        # # Merge on comp_id and atom_id_2
        # merged_2 = atom_df_filtered.merge(bond_df, left_on=['comp_id', 'atom_id'], right_on=['comp_id', 'atom_id_2'], suffixes=('', '_bond_2'))

        # # Concatenate the two merged DataFrames
        # result_df = pd.concat([merged_1, merged_2], ignore_index=True)
        # print(result_df.shape[0])

        # # Apply the function and print the result
        # result_df_final = join_atom_coordinates(result_df, atom_df)

        # # Sort the result_df by comp_id first and then by atom_id
        # result_df_final.sort_values(by=['comp_id', 'atom_id'], inplace=False).reset_index(drop=True)

        # #print(result_df_final.head(15))
        # print(result_df_final.shape[0])