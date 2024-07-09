import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime
import matplotlib.pyplot as plt

# models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# evaluation scheme
from evaluation import *

# Wandb
import wandb


def log_metrics(model_name, metrics, **kwargs):
    # Initialize a new run with a specific name
    wandb.init(project="hydrogen-prediction", name=model_name, reinit=True)
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


def linear_regression(X_train, y_train, X_val, y_val, X_val_coords, type="normal", alpha=0.01):
    if type=="normal":
        model = LinearRegression()
    elif type=="lasso":
        lasso_base = Lasso(alpha=alpha)
        model = MultiOutputRegressor(lasso_base)
    elif type=="ridge":
        ridge_base = Ridge(alpha=alpha)
        model = MultiOutputRegressor(ridge_base)
    else: 
        raise Exception("Linear regression type must be one of normal, lasso, ridge")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return eval_regression(y_val, y_pred, X_val_coords)


def decision_trees(X_train, y_train, X_val, y_val, X_val_coords, type="rf", n_estimators=100, n_jobs=1):
    if type=="rf":
        model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs, random_state=42)
    elif type=="gb":
        base_gb_model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
        model = MultiOutputRegressor(base_gb_model)
    elif type=="xgb":
        model = xgb.XGBRegressor(n_estimators=n_estimators, n_jobs=n_jobs, random_state=42)
    elif type=="cb":
        base_cat_model = CatBoostRegressor(iterations=n_estimators, random_state=42, verbose=0)
        model = MultiOutputRegressor(base_cat_model)
    else: 
        raise Exception("Linear regression type must be one of normal, lasso, ridge")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return eval_regression(y_val, y_pred, X_val_coords)


def support_vector_regressor(X_train, y_train, X_val, y_val, X_val_coords):
    base_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model = MultiOutputRegressor(base_svr_model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return eval_regression(y_val, y_pred, X_val_coords)


def bayesian_regression(X_train, y_train, X_val, y_val, X_val_coords):
    bayesian_ridge_model = BayesianRidge()
    model = MultiOutputRegressor(bayesian_ridge_model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return eval_regression(y_val, y_pred, X_val_coords)


def simple_MLP(X_train, y_train, X_val, y_val, X_val_coords):
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    model = Sequential()
    model.add(Dense(input_dim, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(output_dim))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Set up early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_checkpoint = ModelCheckpoint(f'model_checkpoints/best_model_simple_MLP_{timestamp}.h5', monitor='val_loss', save_best_only=True, mode='min')

    # Train the model with early stopping
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

    y_pred = model.predict(X_val)
    
    return eval_regression(y_val, y_pred, X_val_coords)


def complex_MLP(X_train, y_train, X_val, y_val, X_val_coords):
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(output_dim))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_checkpoint = ModelCheckpoint(f'model_checkpoints/best_model_complex_MLP_{timestamp}.h5', monitor='val_loss', save_best_only=True, mode='min')

    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping, reduce_lr, model_checkpoint])

    y_pred = model.predict(X_val)

    return eval_regression(y_val, y_pred, X_val_coords)


if __name__ == "__main__":
    np.random.seed(42)
    datasets = ['dataset-C4-depth1', 'dataset-O2-depth1', 'dataset-C4-depth2', 'dataset-O2-depth2']
    for dataset_name in datasets:
        train_data_path = f'../data/{dataset_name}/training-validation.npz'
        polynomial_regression = True

        train_data = np.load(train_data_path)
        X_train, y_train = train_data['X_train'], train_data['y_train']

        print("Shape of X: ", X_train.shape)
        print("Shape of y: ", y_train.shape)

        # Split the data to training - validation (not testing yet!)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Extract only the coordinates from X_val
        X_val_coords = extract_relative_coordinates(X_val)

        # For depth 2 datasets, extract the depth 1 coordinates only (this is only needed for computing the angle between H - CENTRAL - NEIGBOR_DEPTH_1)
        if dataset_name[-1] == "2":
            print("Depth 2 If case reached")
            polynomial_regression = False  # also set polynomial regression to false
            X_val_coords_new = []
            if dataset_name[8]=='C':
                for c in X_val_coords:
                    X_val_coords_new.append([c[0], c[1], c[2]])
            elif dataset_name[8]=='O':
                for c in X_val_coords:
                    X_val_coords_new.append([c[0]])            
            X_val_coords = X_val_coords_new
        X_val_coords = np.asarray(X_val_coords)

        # Configurations for wandb
        config = {
            "random_seed": 42,
            "train_data_path": dataset_name
        }

        # Normal Linear Regression
        metrics = linear_regression(X_train, y_train, X_val, y_val, X_val_coords)
        log_metrics("linear_regression", metrics, **config)

        # Lasso Regression
        for alpha in [1, 0.1, 0.01, 0.001]:
            metrics = linear_regression(X_train, y_train, X_val, y_val, X_val_coords, type="lasso", alpha=alpha)
            log_metrics(f"lasso_regression_alpha_{alpha}", metrics, **config)

        # Ridge Regression
        for alpha in [1, 0.1, 0.01, 0.001]:
            metrics = linear_regression(X_train, y_train, X_val, y_val, X_val_coords, type="ridge", alpha=alpha)
            log_metrics(f"ridge_regression_alpha_{alpha}", metrics, **config)

        if polynomial_regression:
            # Polynomial regression of order 2
            poly2 = PolynomialFeatures(degree=2)
            X_train_poly2 = poly2.fit_transform(X_train)
            X_val_poly2 = poly2.transform(X_val)

            # Normal Linear Regression
            metrics = linear_regression(X_train_poly2, y_train, X_val_poly2, y_val, X_val_coords)
            log_metrics("polynomial_linear_regression_order_2", metrics, **config)

            # Lasso Regression
            for alpha in [1, 0.1, 0.01, 0.001]:
                metrics = linear_regression(X_train_poly2, y_train, X_val_poly2, y_val, X_val_coords, type="lasso", alpha=alpha)
                log_metrics(f"polynomial_lasso_regression_order_2_alpha_{alpha}", metrics, **config)

            # Ridge Regression
            for alpha in [1, 0.1, 0.01, 0.001]:
                metrics = linear_regression(X_train_poly2, y_train, X_val_poly2, y_val, X_val_coords, type="ridge", alpha=alpha)
                log_metrics(f"polynomial_ridge_regression_order_2_alpha_{alpha}", metrics, **config)

        # Random Forest
        for n_estimators in [100, 200, 500]:
            metrics = decision_trees(X_train, y_train, X_val, y_val, X_val_coords, type="rf", n_estimators=n_estimators, n_jobs=-1)
            log_metrics(f"random_forest_n_{n_estimators}", metrics, **config)

        # Gradient Boosting
        for n_estimators in [100, 200, 500]:
            metrics = decision_trees(X_train, y_train, X_val, y_val, X_val_coords, type="gb", n_estimators=n_estimators, n_jobs=-1)
            log_metrics(f"gradient_boosting_n_{n_estimators}", metrics, **config)

        # XGBoost
        for n_estimators in [100, 200, 500]:
            metrics = decision_trees(X_train, y_train, X_val, y_val, X_val_coords, type="xgb", n_estimators=n_estimators, n_jobs=-1)
            log_metrics(f"xgboost_n_{n_estimators}", metrics, **config)

        # Catboost
        for n_estimators in [100, 200, 500]:
            metrics = decision_trees(X_train, y_train, X_val, y_val, X_val_coords, type="cb", n_estimators=n_estimators, n_jobs=-1)
            log_metrics(f"catboost_n_{n_estimators}", metrics, **config)

        # Bayesian Regression
        metrics = bayesian_regression(X_train, y_train, X_val, y_val, X_val_coords)
        log_metrics("bayesian_regression", metrics, **config)

        # Simple MLP
        metrics = simple_MLP(X_train, y_train, X_val, y_val, X_val_coords)
        log_metrics("simple_mlp", metrics, **config)

        # Complex MLP
        metrics = complex_MLP(X_train, y_train, X_val, y_val, X_val_coords)
        log_metrics("complex_mlp", metrics, **config)


    # Because SVR takes very long for depth 2, do it at the end
    for dataset_name in datasets:
        train_data_path = f'../data/{dataset_name}/training-validation.npz'
        polynomial_regression = True

        train_data = np.load(train_data_path)
        X_train, y_train = train_data['X_train'], train_data['y_train']

        print("Shape of X: ", X_train.shape)
        print("Shape of y: ", y_train.shape)

        # Split the data to training - validation (not testing yet!)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Extract only the coordinates from X_val
        X_val_coords = extract_relative_coordinates(X_val)

        # For depth 2 datasets, extract the depth 1 coordinates only (this is only needed for computing the angle between H - CENTRAL - NEIGBOR_DEPTH_1)
        if dataset_name[-1] == "2":
            print("Depth 2 If case reached")
            polynomial_regression = False  # also set polynomial regression to false
            X_val_coords_new = []
            if dataset_name[8]=='C':
                for c in X_val_coords:
                    X_val_coords_new.append([c[0], c[1], c[2]])
            elif dataset_name[8]=='O':
                for c in X_val_coords:
                    X_val_coords_new.append([c[0]])            
            X_val_coords = X_val_coords_new
        X_val_coords = np.asarray(X_val_coords)

        # Configurations for wandb
        config = {
            "random_seed": 42,
            "train_data_path": dataset_name
        }

        # Support Vector Regressor
        metrics = support_vector_regressor(X_train, y_train, X_val, y_val, X_val_coords)
        log_metrics("support_vector_regressor", metrics, **config)