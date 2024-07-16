# Hydrogen-Prediction

Hydrogen atoms play a significant role in the analysis and simulation of chemical components. However, large Protein datasets often lack hydrogen, ignoring it’s importance. Therefore, this project’s goal is to develop a tool that is able to recreate the correct position of hydrogen atoms in large molecules.

## Dataset

The dataset used in this project is sourced from the [The Worldwide Protein Data Bank](http://www.wwpdb.org/data/ccd). This comprehensive database provides detailed information on protein molecules, including their atomic composition and bonding structures. Some key statistics:
- 41440 protein molecules
- 286717 Carbon atoms
- 53803 Oxygen atoms

## Project Structure

```bash
.
├── data/            		        
├── img/             		        
├── src/           		            
	├── model_checkpoints/          # Weights of best MLPs
    ├── calc_wasserstein.py  	    # Method to compute min wasserstein distance  	
    ├── config.py  				    # Configuration parameters		
	├── data_preprocessing.py       # Data preprocessing + generation 
	├── model_selection.ipynb  		# Selection of best models after training		 
	├── models.py                   # Model definitions
	├── test_models.ipynb        	# (optional) Testing of 3 best models per dataset		 
	├── test_models.py  			# Testing of 3 best models per dataset			 
	├── train_models.ipynb        	# (optional) Training models per dataset		 
	├── train_models.py      		# Training models per dataset			 
	└── utils.py                    # Utility methods
├── .gitignore
├── README.md
├── requirements.txt
```

## Workflow

### Prerequisities

1. We recommend making a new virtual environment (e.g., using Anaconda). For this project, we used Python 3.11.9. For example, you can create a new Anaconda environment called "hydrogen" with Python version 3.11.9 using the following command:
```
conda create --name hydrogen python=3.11.9
```
Activate the environment via
```
conda activate hydrogen
```
2. Install the dependencies via
```
pip install -r requirements.txt
```
3. Make a new folder called ```data/``` on the same level as ```src/```, e.g. via
```
mkdir data
```
4. Download ```components.cif``` from [Worldwide Protein Data Bank](http://www.wwpdb.org/data/ccd) and put it into the folder ```data/```

### Data Preprocessing and Data Generation

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
2. Run ```data_preprocessing.py```. It will put the whole dataset, training-validation, testing dataset (and maximum length of the input without zeropadding and encoding) into the path specified by the parameters that end with ```_save_path```.
```
python data_preprocessing.py
```

3. Train Linear regression models with notebook ```linear_regression.ipynb``` and tree-based models with notebook ```bagging_and_boosting.ipynb```

### Jupyter notebooks
You can work on this project using an Integrated Development Environment (IDE) that supports Jupyter notebooks, such as VSCode or PyCharm, or you can use Jupyter Notebook itself:
```
jupyter notebook
```
If there appears a blank webpage, press "CTRL+F5" to force the page reloading.

## References
- Kunzmann, P., Anter, J. M., & Hamacher, K. (2022). <a href="https://link.springer.com/article/10.1186/s13015-022-00215-x">Adding hydrogen atoms to molecular models via fragment superimposition.</a> Algorithms for Molecular Biology, 17(1), 7.

## Todos

For poster:
- Make visualization of our workflow?

Implementation:
- Make train_models.py and test_models.py work also with connections data structure
- make a check for model_checkpoints/ folder in test_models.py (train_models.py must be run before test_models.py)