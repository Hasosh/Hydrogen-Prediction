from sklearn.multioutput import MultiOutputRegressor
from datetime import datetime
import os

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

# Config File
from config import Config


def linear_regression(X_train, y_train, X_val, type="normal", alpha=0.01):
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
    return y_pred


def decision_trees(X_train, y_train, X_val, type="rf", n_estimators=100, n_jobs=1):
    if type=="rf":
        model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs, random_state=Config.random_state)
    elif type=="gb":
        base_gb_model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=Config.random_state)
        model = MultiOutputRegressor(base_gb_model)
    elif type=="xgb":
        model = xgb.XGBRegressor(n_estimators=n_estimators, n_jobs=n_jobs, random_state=Config.random_state)
    elif type=="cb":
        base_cat_model = CatBoostRegressor(iterations=n_estimators, random_state=Config.random_state, verbose=0)
        model = MultiOutputRegressor(base_cat_model)
    else: 
        raise Exception("Linear regression type must be one of normal, lasso, ridge")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return y_pred


def support_vector_regressor(X_train, y_train, X_val):
    base_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model = MultiOutputRegressor(base_svr_model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return y_pred


def bayesian_regression(X_train, y_train, X_val):
    bayesian_ridge_model = BayesianRidge()
    model = MultiOutputRegressor(bayesian_ridge_model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return y_pred


def create_model_checkpoint_dir(directory="model_checkpoints"):
    if not os.path.exists(directory):
        os.makedirs(directory)


def simple_MLP(X_train, y_train, X_val):
    create_model_checkpoint_dir()

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
    return y_pred


def complex_MLP(X_train, y_train, X_val):
    create_model_checkpoint_dir()

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
    return y_pred