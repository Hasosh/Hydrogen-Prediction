# Hydrogen-Prediction

Hydrogen atoms play a significant role in the analysis and simulation of chemical components. However, large Protein datasets often lack hydrogen, ignoring it’s importance. Therefore, this project’s goal is to develop a tool that is able to recreate the correct position of hydrogen atoms in large molecules.

Here you can find our [poster](https://github.com/Hasosh/Hydrogen-Prediction/blob/main/doc/poster.pdf) that summarizes our main results of the project.

## Dataset

The dataset used in this project is the Chemical Component Dictionary sourced from the [The Worldwide Protein Data Bank](http://www.wwpdb.org/data/ccd). This comprehensive database provides detailed information on protein molecules, including their atomic composition and bonding structures. Some key statistics:
- 41440 protein molecules
- 286717 Carbon atoms
- 53803 Oxygen atoms

## Project Structure

```bash
.
├── data/                         # Directory for data
├── img/                          # Directory for images
├── src/                          # Source directory
│   ├── model_checkpoints/        # Weights of best MLPs
│   ├── notebooks/                # Optional notebooks (no new functionality)
│   ├── calc_wasserstein.py       # Compute min. Wasserstein distance
│   ├── config.py                 # Configuration parameters
│   ├── data_preprocessing.py     # Data preprocessing and generation
│   ├── model_selection.py        # Selection of best models after training
│   ├── models.py                 # Model definitions
│   ├── test_models.py            # Testing of 3 best models per dataset
│   ├── train_models.py           # Training models per dataset
│   └── utils.py                  # Utility methods
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
For more information, please refer to [Anacondas User Guide](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

2. Install the dependencies via
```
pip install -r requirements.txt
```
3. Make a new folder called ```data/``` on the same level as ```src/```, e.g. via
```
mkdir data
```
4. Download ```components.cif``` from [Worldwide Protein Data Bank](http://www.wwpdb.org/data/ccd) and put it into the folder ```data/```.

### Notebooks
This project includes optional notebooks designed to aid in debugging and understanding the Python scripts. While they do not introduce new functionality, they can be very helpful for deeper insight. Note that for the notebooks, you may need to adapt (relative) paths in the code.

You can open these notebooks using an Integrated Development Environment (IDE) that supports notebooks, such as VSCode or PyCharm, or by using Jupyter Notebook itself.

To install Jupyter Notebook, refer to the following resources:
- https://jupyter.org/install
- https://stackoverflow.com/questions/58068818/how-to-use-jupyter-notebooks-in-a-conda-environment

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
    - Set the remaining parameters w.r.t. your preference.
2. Run ```data_preprocessing.py```. It will put the whole dataset, training-validation set, testing set (and connections data structure) into the path specified by the parameters that end with ```_save_path```. When running it the first time, it may take some while (around 2 hours). After a successful run, you can set the parameters ```make_bond_df, make_atom_df, make_atom_df_extended, make_preprocessed_data``` to ```False``` as they need to be computed one time only.
```
python data_preprocessing.py
```
3. (optional) Note that you may need to repeat steps 1. and 2. several times with different configuration parameters to generate different datasets.

### Training

1. Specify the needed parameters in ```config.py```:
    - use_wandb 
    - wandb_project_name 
    - datasets
    - alpha_values 
    - do_polynomial_regression 
    - estimator_values 
    - n_jobs

2. If ```use_wandb = True```, then logged metrics are uploaded to [WandB](https://wandb.ai/site). Note that you cannot proceed with model selection if you do not use WandB.

For that you need to create a Wandb account and login to your account via writing 
```
wandb login
```
into the command line and pasting your WandB API key. For more information, please click [here](https://docs.wandb.ai/quickstart).

3. Run ```train_models.py```. 
```
python train_models.py
```
It will train various models (see ```models.py``` for more information) and log the metrics MSE, R2, average cosine similarity, average binding length, bond angle wasserstein distance (and upload them to WandB).

### Model Selection

1. Specify the needed parameters in ```config.py``` (```wandb_entity``` can be found out through WandB, ```wandb_project_name``` should be equal to what you specified during training phase):
    - wandb_entity     
    - wandb_project_name 

2. Run ```model_selection.py```: 
```
python model_selection.py
```
It computes metrics and identifies the best models per dataset through a ranking. The plots are saved in ```img/training/```. 

The best models are tested in ```test_models.py``` later.

### Testing

1. Specify the parameters in ```config.py```:
    - use_wandb 
    - wandb_project_name 
    - datasets
    - n_jobs

We recommend to use a different WandB project name for testing than for training to separate the results better. The other parameters can remain unchanged.

2. Set the model checkpoints in ```test_models.py``` (look for variable ```checkpoint_path```) if MLP models belong to one of the best models. In the folder ```src/model_checkpoints``` there can be found checkpoints of the MLP models where they performed best on the validation set.

3. Please review the ranking plots from the model selection phase. Choose the top three models and write the corresponding code to test them in ```test_models.py```. You need to do that per dataset and consequently set the if clause correctly. You can use the provided code as a reference. 

4. If ```use_wandb = True```, then logged metrics are uploaded to [WandB](https://wandb.ai/site).

5. Run ```test_models.py```. 
```
python test_models.py
```
It tests the best 3 models per dataset. For all models the metrics MSE, R2, average cosine similarity, average binding length, bond angle wasserstein distance are logged (and uploaded to WandB). In addition, distribution plots between ground truth and predicted are made for bond angles, bond length and dihedral angles. You can download the plots from WandB.

## Related Work
- Kunzmann, P., Anter, J. M., & Hamacher, K. (2022). <a href="https://link.springer.com/article/10.1186/s13015-022-00215-x">Adding hydrogen atoms to molecular models via fragment superimposition.</a> Algorithms for Molecular Biology, 17(1), 7.