import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import pickle

# Config File
from config import Config

# models
from models import *

# evaluation scheme
from utils import *

# Wandb
import wandb


def log_metrics(model_name, metrics, **kwargs):
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
        wandb.finish()


if __name__ == "__main__":
    np.random.seed(Config.random_state)
    for dataset_name in Config.datasets:
        train_data_path = f'../data/{dataset_name}/training-validation.npz'

        train_data = np.load(train_data_path)
        X_train, y_train = train_data['X_train'], train_data['y_train']

        print("Shape of X: ", X_train.shape)
        print("Shape of y: ", y_train.shape)

        with open(f'../data/{dataset_name}/connections.pkl', 'rb') as f:
            connections = pickle.load(f)
        connections_train, _ = connections  # we just need connections_train

        # Split the data to training - validation (not testing yet!)
        X_train, X_val, y_train, y_val, connections_train, connections_val = train_test_split(X_train, y_train, connections_train, test_size=Config.train_test_split_ratio, random_state=Config.random_state)

        # -------------------------------------------------
        # Extract (relative) coordinates from X_test
        # -------------------------------------------------

        # Extract only the coordinates from X_val
        X_val_coords = extract_relative_coordinates_of_depth1(X_val, connections_val)

        # Configurations for wandb
        config = {
            "random_seed": Config.random_state,
            "train_data_path": dataset_name
        }

        # Normal Linear Regression
        print("Training Normal Linear Regression")
        y_pred = linear_regression(X_train, y_train, X_val)
        metrics = eval_regression(y_val, y_pred, X_val_coords)
        log_metrics("linear_regression", metrics, **config)

        # Lasso Regression
        print("Training Lasso Regression")
        for alpha in Config.alpha_values:
            y_pred = linear_regression(X_train, y_train, X_val, type="lasso", alpha=alpha)
            metrics = eval_regression(y_val, y_pred, X_val_coords)
            log_metrics(f"lasso_regression_alpha_{alpha}", metrics, **config)

        # Ridge Regression
        print("Training Ridge Regression")
        for alpha in Config.alpha_values:
            y_pred = linear_regression(X_train, y_train, X_val, type="ridge", alpha=alpha)
            metrics = eval_regression(y_val, y_pred, X_val_coords)
            log_metrics(f"ridge_regression_alpha_{alpha}", metrics, **config)

        if Config.do_polynomial_regression and dataset_name[-1] != "2":
            # Polynomial regression of order 2
            print("Computing polynomial features")
            poly2 = PolynomialFeatures(degree=2)
            X_train_poly2 = poly2.fit_transform(X_train)
            X_val_poly2 = poly2.transform(X_val)

            # Normal Linear Regression
            print("Training Normal Linear Regression")
            y_pred = linear_regression(X_train_poly2, y_train, X_val_poly2)
            metrics = eval_regression(y_val, y_pred, X_val_coords)
            log_metrics("polynomial_linear_regression_order_2", metrics, **config)

            # Lasso Regression
            print("Training Lasso Regression")
            for alpha in Config.alpha_values:
                y_pred = linear_regression(X_train_poly2, y_train, X_val_poly2, type="lasso", alpha=alpha)
                metrics = eval_regression(y_val, y_pred, X_val_coords)
                log_metrics(f"polynomial_lasso_regression_order_2_alpha_{alpha}", metrics, **config)

            # Ridge Regression
            print("Training Ridge Regression")
            for alpha in Config.alpha_values:
                y_pred = linear_regression(X_train_poly2, y_train, X_val_poly2, type="ridge", alpha=alpha)
                metrics = eval_regression(y_val, y_pred, X_val_coords)
                log_metrics(f"polynomial_ridge_regression_order_2_alpha_{alpha}", metrics, **config)

        # Random Forest
        print("Training Random Forest Regressor")
        for n_estimators in Config.estimator_values:
            y_pred = decision_trees(X_train, y_train, X_val, type="rf", n_estimators=n_estimators, n_jobs=Config.n_jobs)
            metrics = eval_regression(y_val, y_pred, X_val_coords)
            log_metrics(f"random_forest_n_{n_estimators}", metrics, **config)

        # Gradient Boosting
        print("Training Gradient Boosting Regressor")
        for n_estimators in Config.estimator_values:
            y_pred = decision_trees(X_train, y_train, X_val, type="gb", n_estimators=n_estimators, n_jobs=Config.n_jobs)
            metrics = eval_regression(y_val, y_pred, X_val_coords)
            log_metrics(f"gradient_boosting_n_{n_estimators}", metrics, **config)

        # XGBoost
        print("Training XGBoost Regressor")
        for n_estimators in Config.estimator_values:
            y_pred = decision_trees(X_train, y_train, X_val, type="xgb", n_estimators=n_estimators, n_jobs=Config.n_jobs)
            metrics = eval_regression(y_val, y_pred, X_val_coords)
            log_metrics(f"xgboost_n_{n_estimators}", metrics, **config)

        # Catboost
        print("Training Catboost Regressor")
        for n_estimators in Config.estimator_values:
            y_pred = decision_trees(X_train, y_train, X_val, type="cb", n_estimators=n_estimators, n_jobs=Config.n_jobs)
            metrics = eval_regression(y_val, y_pred, X_val_coords)
            log_metrics(f"catboost_n_{n_estimators}", metrics, **config)

        # Support Vector Regressor
        print("Training Support Vector Regressor")
        y_pred = support_vector_regressor(X_train, y_train, X_val)
        metrics = eval_regression(y_val, y_pred, X_val_coords)
        log_metrics("support_vector_regressor", metrics, **config)

        # Bayesian Regression
        print("Training Bayesian Regressor")
        y_pred = bayesian_regression(X_train, y_train, X_val)
        metrics = eval_regression(y_val, y_pred, X_val_coords)
        log_metrics("bayesian_regression", metrics, **config)

        # Simple MLP
        print("Training simple MLP")
        y_pred = simple_MLP(X_train, y_train, X_val)
        metrics = eval_regression(y_val, y_pred, X_val_coords)
        log_metrics("simple_mlp", metrics, **config)

        # Complex MLP
        print("Training complex MLP")
        y_pred = complex_MLP(X_train, y_train, X_val)
        metrics = eval_regression(y_val, y_pred, X_val_coords)
        log_metrics("complex_mlp", metrics, **config)

