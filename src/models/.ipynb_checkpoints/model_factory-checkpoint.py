import numpy as np
import logging
import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from utils.helpers import prepare_data


# Logging config
logging.basicConfig(level=logging.INFO)

# === GridSearch Model Constructor===
def build_model(model_name, X_train, y_train, model_params=None):
    """
    Constructs and trains a model based on the specified model name.

    Args:
        model_name (str): The name of the model to build and train (e.g., 'neural_network', 'svm').
        X_train (numpy.ndarray): The training input features.
        y_train (numpy.ndarray): The training target labels.
        model_params (dict, optional): Dictionary with model instances and hyperparameters for GridSearch.
            If None, default parameters will be used.

    Returns:
        object: The trained model.

    Raises:
        ValueError: If the specified model name is not supported.
    """
    model_name = model_name.lower()

    if model_name == "neural_network":
        input_dim = X_train.shape[1]
        model = NeuralNetworkModel(input_dim=input_dim, hidden_layer_sizes=(100,))
        model.fit(X_train, y_train, epochs=100, batch_size=32, lr=0.001)
        return model

    # Default GridSearch Models Dictionary
    default_params = {
            "svm": (
                SVC(probability=True),
                {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto']}
            ),
            "random_forest": (
                RandomForestClassifier(),
                {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 10]}
            ),
            "logistic_regression": (
                LogisticRegression(),
                {'C': [0.01, 0.1, 1], 'solver': ['liblinear', 'lbfgs']}
            ),
            "knn": (
                KNeighborsClassifier(),
                {'n_neighbors': [3, 7], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
            ),
            "adaboost": (
                AdaBoostClassifier(),
                {'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 1.0, 10.0]}
            ),
            "naive_bayes": (
                GaussianNB(),
                {'var_smoothing': [1e-8, 1e-7, 1e-6, 1e-5]}
            )
        }
    
    if model_params is None:
        model_params = default_params
        
    if model_name not in model_params:
        raise ValueError(f"Model '{model_name}' is not supported.")

    base_model, param_grid = model_params[model_name]

    logging.info(f"[{model_name.upper()}] 📚 UFC GridSearchCV Training 📚...")
    
    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', error_score='raise', verbose = 3)
    grid_search.fit(X_train, y_train)

    time.sleep(1)
    
    logging.info(f"[{model_name.upper()}] 🔍 Best Score: {grid_search.best_score_:.4f}")
    logging.info(f"[{model_name.upper()}] 🔍 Best Params: {grid_search.best_params_}")

    return grid_search

# === Model Selection ===
def model_factory(model_name, data, model_params=None):
    """
    Selects and builds a model based on the specified model name and training data.
    
    Args:
        model_name (str): The name of the model to be selected.
        data (pandas.DataFrame): The training dataset, which must include a 'label' column.
    
    Returns:
        object: The trained model.
    
    Raises:
        ValueError: If the 'label' column is missing from the training data.
    """
    if 'label' not in data.columns:
        raise ValueError("The dataframe must contain a 'label' column.")

    X_train, X_test, y_train, y_test = prepare_data(data)

    return build_model(model_name, X_train, y_train, model_params)




