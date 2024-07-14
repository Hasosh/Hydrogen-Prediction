import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io
import pickle
from PIL import Image
from tensorflow.keras.models import load_model

# Config File
from config import Config

# models
from models import *

# evaluation scheme
from utils import *

# Wandb
import wandb

# Wasserstein distance
from calc_wasserstein import *


def plot_first_degree_bond_angles_distributions(y_test, y_pred, X_test_coords, dataset_name, model_name):
    # Calculate all angles
    all_angles_test = all_angles_H_central_neighbor(X_test_coords, y_test)
    all_angles_pred = all_angles_H_central_neighbor(X_test_coords, y_pred)

    # Calculate Wasserstein distance
    distance = calculate_wasserstein_distance(all_angles_test, all_angles_pred, bins=180, range=(0, 180))
    print(f'Wasserstein distance for first degree bond angle distributions: {distance}')

    # Create figure
    fig, ax = plt.subplots()

    # Plot histograms
    ax.hist(all_angles_test, bins=180, density=True, alpha=0.7, label='all_angles_test')
    ax.hist(all_angles_pred, bins=180, density=True, alpha=0.5, color="red", label='all_angles_pred')
    ax.set_title(f"first degree bond angles distribution for {dataset_name}, model: {model_name}")
    ax.set_xlabel("Degree (Angle)")
    ax.set_ylabel("Density")

    # Add Wasserstein distance text
    ax.text(0.55, 0.75, f'Wasserstein distance: {distance:.4f}', 
             horizontalalignment='right', 
             verticalalignment='top', 
             transform=plt.gca().transAxes)

    # Add legend
    ax.legend()

    # Show plot
    #plt.show()

    # Save figure to a bytes buffer with bbox_inches='tight'
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    plt.close(fig)  # Close the figure to free memory

    return buf


def plot_bindinglength_distribution(y_test, y_pred, dataset_name, model_name):
    # Calculate binding lengths
    all_bindinglengths_test = bindungslängen(y_test)
    all_bindinglengths_pred = bindungslängen(y_pred)

    # Calculate Wasserstein distance
    distance = calculate_wasserstein_distance(all_bindinglengths_test, all_bindinglengths_pred, bins=100, range=(0, 1.5))
    print(f'Wasserstein distance for bindings length distributions: {distance}')

    # Create figure
    fig, ax = plt.subplots()

    # Plot histograms
    ax.hist(all_bindinglengths_test, bins=100, density=True, alpha=0.7, label='all_bindinglengths_test')
    ax.hist(all_bindinglengths_pred, bins=100, density=True, alpha=0.5, color="red", label='all_bindinglengths_pred')
    ax.set_title(f"Binding length distribution for {dataset_name}, model: {model_name}")
    ax.set_xlabel("Binding length")
    ax.set_ylabel("Density")

    # Add Wasserstein distance text
    ax.text(0.55, 0.75, f'Wasserstein distance: {distance:.4f}', 
            horizontalalignment='right', 
            verticalalignment='top', 
            transform=ax.transAxes)

    # Add legend
    ax.legend()

    # Show plot
    #plt.show()

    # Save figure to a bytes buffer with bbox_inches='tight'
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    plt.close(fig)  # Close the figure to free memory

    return buf


def plot_dihedral_distribution(y_test, y_pred, X_test, connections_test, dataset_name, model_name):
    # Calculate dihedral angles for test set
    all_dihedral_angles_test = []
    for i, tuples_list_raw in enumerate(connections_test):
        neighbors_dict = find_neighbors(tuples_list_raw, -1, 2)
        for key, tuples_list in neighbors_dict.items():
            fixed_index = int(key[-1] / 5 * 11 + 8)
            for tuple in tuples_list:
                varying_index = int(tuple[-1] / 5 * 11 + 8)
                dihedral_angle = compute_dihedral(y_test[i], X_test[i][fixed_index:fixed_index + 3], [0, 0, 0], X_test[i][varying_index:varying_index + 3])
                all_dihedral_angles_test.append(dihedral_angle)

    # Calculate dihedral angles for predicted set
    all_dihedral_angles_pred = []
    for i, tuples_list_raw in enumerate(connections_test):
        neighbors_dict = find_neighbors(tuples_list_raw, -1, 2)
        for key, tuples_list in neighbors_dict.items():
            fixed_index = int(key[-1] / 5 * 11 + 8)
            for tuple in tuples_list:
                varying_index = int(tuple[-1] / 5 * 11 + 8)
                dihedral_angle = compute_dihedral(y_pred[i], X_test[i][fixed_index:fixed_index + 3], [0, 0, 0], X_test[i][varying_index:varying_index + 3])
                all_dihedral_angles_pred.append(dihedral_angle)

    # Calculate Wasserstein distance
    distance = calculate_wasserstein_distance(all_dihedral_angles_test, all_dihedral_angles_pred, bins=180, range=(0, 180))
    print(f'Wasserstein distance for dihedral angle distributions: {distance}')

    # Create figure
    fig, ax = plt.subplots()

    # Plot histograms
    ax.hist(all_dihedral_angles_test, bins=180, density=True, alpha=0.7, label='all_dihedral_angles_test')
    ax.hist(all_dihedral_angles_pred, bins=180, density=True, alpha=0.5, color="red", label='all_dihedral_angles_pred')
    ax.set_title(f"Dihedral angle distributions for {dataset_name}, model: {model_name}")
    ax.set_xlabel("Degree (Angle)")
    ax.set_ylabel("Density")

    # Add Wasserstein distance text
    ax.text(0.55, 0.75, f'Wasserstein distance: {distance:.4f}', 
            horizontalalignment='right', 
            verticalalignment='top', 
            transform=ax.transAxes)

    # Add legend
    ax.legend()

    # Show plot
    #plt.show()

    # Save figure to a bytes buffer with bbox_inches='tight'
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    plt.close(fig)  # Close the figure to free memory

    return buf


def make_plots(y_test, y_pred, X_test, X_test_coords, dataset_name, model_name, connections_test=None, depth=1):
    buffer = []
    # Make plot for first degree bond angle distributions
    buf = plot_first_degree_bond_angles_distributions(y_test, y_pred, X_test_coords, dataset_name, model_name)
    buffer.append(buf)

    # Make plot for first degree bond angle distributions
    buf = plot_bindinglength_distribution(y_test, y_pred, dataset_name, model_name)
    buffer.append(buf)

    if depth > 1:
        # Make plot for dihedral distributions
        buf = plot_dihedral_distribution(y_test, y_pred, X_test, connections_test, dataset_name, model_name)
        buffer.append(buf)

    return buffer


def log_metrics(model_name, metrics, plot_buf=None, **kwargs):
    if Config.use_wandb:
        # Initialize a new run with a specific name
        wandb.init(project=Config.wandb_project_name, name=model_name, reinit=True)
        wandb.config.update(kwargs)
        wandb.log({
            "model_name": model_name,
            "MSE": metrics[0],
            "R2": metrics[1],
            "avg_cosine_similarity": metrics[2],
            "avg_binding_length": metrics[3],
            "tri-angle_wasserstein_distance": metrics[4]
        })
        if plot_buf:
            for buf in plot_buf:
                buf.seek(0)
                img = Image.open(buf)
                wandb.log({"Plot": wandb.Image(img)})
        wandb.finish()


if __name__ == "__main__":
    np.random.seed(Config.random_state)
    for dataset_name in Config.datasets:
        depth = int(dataset_name[-1])
        # -------------------------------------------------
        # Load training data
        # -------------------------------------------------

        train_data_path = f'../data/{dataset_name}/training-validation.npz'

        train_data = np.load(train_data_path)
        X_train, y_train = train_data['X_train'], train_data['y_train']

        print("Shape of X_train: ", X_train.shape)
        print("Shape of y_train: ", y_train.shape)

        # Split the data to training - validation (not testing yet!), this is to make it comparable to train_models.py
        X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.2, random_state=Config.random_state)

        # -------------------------------------------------
        # Load testing data
        # -------------------------------------------------

        test_data_path = f'../data/{dataset_name}/testing.npz'

        test_data = np.load(test_data_path)
        X_test, y_test = test_data['X_test'], test_data['y_test']

        print("Shape of X_test: ", X_test.shape)
        print("Shape of y_test: ", y_test.shape)

        # -------------------------------------------------
        # Load connections
        # -------------------------------------------------

        if depth == 2:
            with open(f'../data/{dataset_name}/connections.pkl', 'rb') as f:
                connections = pickle.load(f)
            _, connections_test = connections  # we just need connections_test

        # -------------------------------------------------
        # Extract (relative) coordinates from X_test
        # -------------------------------------------------

        # Extract only the coordinates from X_test
        X_test_coords = extract_relative_coordinates(X_test)

        # For depth 2 datasets, extract the depth 1 coordinates only (this is only needed for computing the angle between H - CENTRAL - NEIGBOR_DEPTH_1)
        if depth == 2:
            print("Depth 2 If case reached")
            X_test_coords_new = []
            if dataset_name[8]=='C':
                for c in X_test_coords:
                    X_test_coords_new.append([c[0], c[1], c[2]])
            elif dataset_name[8]=='O':
                for c in X_test_coords:
                    X_test_coords_new.append([c[0]])            
            X_test_coords = X_test_coords_new
        X_test_coords = np.asarray(X_test_coords)

        # Configurations for wandb
        config = {
            "random_seed": Config.random_state,
            "train_data_path": dataset_name
        }

        # -------------------------------------------------
        # Apply test set to the best 3 models (on training performance) per dataset
        # -------------------------------------------------

        if dataset_name == 'dataset-C4-depth1':
            # 1. Support Vector Regressor
            model_name = "Support Vector Regressor"
            y_pred = support_vector_regressor(X_train, y_train, X_test)
            metrics = eval_regression(y_test, y_pred, X_test_coords)
            
            # make distribution plots
            buffer = make_plots(y_test, y_pred, X_test, X_test_coords, dataset_name, model_name, depth=depth)

            # Upload to wandb
            log_metrics(model_name, metrics, plot_buf=buffer, **config)


            # 2. Simple MLP
            model_name = "Simple-MLP"
            model = load_model(f'model_checkpoints/best_model_simple_MLP_20240709-024225.h5')
            y_pred = model.predict(X_test)
            metrics = eval_regression(y_test, y_pred, X_test_coords)

            # make distribution plots
            buffer = make_plots(y_test, y_pred, X_test, X_test_coords, dataset_name, model_name, depth=depth)

            # Upload to wandb
            log_metrics(model_name, metrics, plot_buf=buffer, **config)


            # 3. Catboost 500
            model_name = "Catboost-500"
            y_pred = decision_trees(X_train, y_train, X_test, type="cb", n_estimators=500, n_jobs=Config.n_jobs)
            metrics = eval_regression(y_test, y_pred, X_test_coords)

            # make distribution plots
            buffer = make_plots(y_test, y_pred, X_test, X_test_coords, dataset_name, model_name, depth=depth)

            # Upload to wandb
            log_metrics(model_name, metrics, plot_buf=buffer, **config)

        elif dataset_name == 'dataset-C4-depth2':
            # 1. Catboost 500
            model_name = "Catboost-500"
            y_pred = decision_trees(X_train, y_train, X_test, type="cb", n_estimators=500, n_jobs=Config.n_jobs)
            metrics = eval_regression(y_test, y_pred, X_test_coords)

            # make distribution plots
            buffer = make_plots(y_test, y_pred, X_test, X_test_coords, dataset_name, model_name, connections_test=connections_test, depth=depth)

            # Upload to wandb
            log_metrics(model_name, metrics, plot_buf=buffer, **config)


            # 2. Random Forest 500
            model_name = "RandomForest-500"
            y_pred = decision_trees(X_train, y_train, X_test, type="rf", n_estimators=500, n_jobs=Config.n_jobs)
            metrics = eval_regression(y_test, y_pred, X_test_coords)

            # make distribution plots
            buffer = make_plots(y_test, y_pred, X_test, X_test_coords, dataset_name, model_name, connections_test=connections_test, depth=depth)

            # Upload to wandb
            log_metrics(model_name, metrics, plot_buf=buffer, **config)


            # 3. Simple MLP
            model_name = "Simple-MLP"
            model = load_model(f'model_checkpoints/best_model_simple_MLP_20240709-031301.h5')
            y_pred = model.predict(X_test)
            metrics = eval_regression(y_test, y_pred, X_test_coords)

            # make distribution plots
            buffer = make_plots(y_test, y_pred, X_test, X_test_coords, dataset_name, model_name, connections_test=connections_test, depth=depth)

            # Upload to wandb
            log_metrics(model_name, metrics, plot_buf=buffer, **config)

        elif dataset_name == 'dataset-O2-depth1':
            # 1. Catboost 200
            model_name = "Catboost-200"
            y_pred = decision_trees(X_train, y_train, X_test, type="cb", n_estimators=200, n_jobs=Config.n_jobs)
            metrics = eval_regression(y_test, y_pred, X_test_coords)

            # make distribution plots
            buffer = make_plots(y_test, y_pred, X_test, X_test_coords, dataset_name, model_name, depth=depth)

            # Upload to wandb
            log_metrics(model_name, metrics, plot_buf=buffer, **config)
            

            # 2. Gradient boosting 500
            model_name = "Gradientboost-500"
            y_pred = decision_trees(X_train, y_train, X_test, type="gb", n_estimators=500, n_jobs=Config.n_jobs)
            metrics = eval_regression(y_test, y_pred, X_test_coords)

            # make distribution plots
            buffer = make_plots(y_test, y_pred, X_test, X_test_coords, dataset_name, model_name, depth=depth)

            # Upload to wandb
            log_metrics(model_name, metrics, plot_buf=buffer, **config)


            # 3. Simple MLP
            model_name = "Simple-MLP"
            model = load_model(f'model_checkpoints/best_model_simple_MLP_20240709-025014.h5')
            y_pred = model.predict(X_test)
            metrics = eval_regression(y_test, y_pred, X_test_coords)

            # make distribution plots
            buffer = make_plots(y_test, y_pred, X_test, X_test_coords, dataset_name, model_name, depth=depth)

            # Upload to wandb
            log_metrics(model_name, metrics, plot_buf=buffer, **config)

        elif dataset_name == 'dataset-O2-depth2':
            # 1. Simple MLP
            model_name = "Simple-MLP"
            model = load_model(f'model_checkpoints/best_model_simple_MLP_20240709-032158.h5')
            y_pred = model.predict(X_test)
            metrics = eval_regression(y_test, y_pred, X_test_coords)

            # make distribution plots
            buffer = make_plots(y_test, y_pred, X_test, X_test_coords, dataset_name, model_name, connections_test=connections_test, depth=depth)

            # Upload to wandb
            log_metrics(model_name, metrics, plot_buf=buffer, **config)           
           
            
            # 2. Catboost 500
            model_name = "Catboost-500"
            y_pred = decision_trees(X_train, y_train, X_test, type="cb", n_estimators=500, n_jobs=Config.n_jobs)
            metrics = eval_regression(y_test, y_pred, X_test_coords)

            # make distribution plots
            buffer = make_plots(y_test, y_pred, X_test, X_test_coords, dataset_name, model_name, connections_test=connections_test, depth=depth)

            # Upload to wandb
            log_metrics(model_name, metrics, plot_buf=buffer, **config)


            # 3. Support Vector Regressor
            model_name = "Support Vector Regressor"
            y_pred = support_vector_regressor(X_train, y_train, X_test)
            metrics = eval_regression(y_test, y_pred, X_test_coords)
            
            # make distribution plots
            buffer = make_plots(y_test, y_pred, X_test, X_test_coords, dataset_name, model_name, connections_test=connections_test, depth=depth)

            # Upload to wandb
            log_metrics(model_name, metrics, plot_buf=buffer, **config)



