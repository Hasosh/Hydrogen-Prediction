# Hydrogen-Prediction

## Prerequisities

1. Install the dependencies via
```
pip install -r requirements.txt
```
2. Make a new folder called ```data/``` on the same level as ```src/```
3. Download ```components.cif``` from [Worldwide Protein Data Bank](http://www.wwpdb.org/data/ccd) and put it into the folder ```data/```

## Workflow

1. Specify the parameters in ```config.py```
    - If you do not already have the dataframe ```bond_df, atom_df, atom_df_extended```, set the parameters ```make_bond_df, make_atom_df, make_atom_df_extended``` to ```True```, else to ```False```.
    - If you do not already have preprocessed the data (i.e. created adjacency matrices etc.), set the parameter ```make_preprocessed_data``` to ```True```, else to ```False```.
    - For the C dataset set the parameters
        - num_hydrogens = 1
        - central_atom = 'C'
        - neighbor_depth = 1 / 2
        - num_neighbors_to_centralatom = 4
    - For the O dataset, set the parameters
        - num_hydrogens = 1
        - central_atom = 'O'
        - neighbor_depth = 1 / 2
        - num_neighbors_to_centralatom = 2 
    - Set the remaining parameters w.r.t. your preference
2. Run ```data_preprocessing2.py``` (not ```data_preprocessing.py```). It will put the whole dataset, training-validation, testing dataset (and maximum length of the input without zeropadding and encoding) into the path specified by the parameters that end with ```_save_path```.
```
python data_preprocessing2.py
```


3. Train Linear regression models with notebook ```linear_regression.ipynb``` and tree-based models with notebook ```bagging_and_boosting.ipynb```

## Todos

For poster:
- Make chemical plots for the introduction using Pymol
- Make visualization of our workflow

For implementation:
- Train different models on the above parameter case and look at their metrics using ```evaluation.py```
- Look at the variability of the data in a notebook using the methods in ```evaluation.py```