from gemmi import cif
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


if __name__ == "__main__":
    doc = cif.read_file("../data/components.cif")
    print('Numver of compounds in dataset: '+ str(len(doc)))

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