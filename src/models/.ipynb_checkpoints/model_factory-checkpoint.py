import numpy as np
import logging
import time
from sklearn.model_selection import GridSearchCV
from utils.helpers import *
from .config import default_params
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Logging config
logging.basicConfig(level=logging.INFO)

# === Supported Models List ===
SUPPORTED_MODELS = list(default_params.keys())

# === GridSearch Model Constructor ===
def build_model(model_name, X_train, y_train, model_params=None):
    """
    Constructs and trains a model based on the specified model name.

    Args:
        model_name (str): The name of the model to build and train (e.g., 'svm', 'random_forest').
        X_train (numpy.ndarray): The training input features.
        y_train (numpy.ndarray): The training target labels.
        model_params (dict, optional): Dictionary with model instances and hyperparameters for GridSearch.
            If None, default parameters will be used.

    Returns:
        object: The trained model.

    Raises:
        ValueError: If the specified model name is not supported.
    """
    
    if model_params is None:
        model_params = default_params

    if model_name not in model_params:
        raise ValueError(
            f"Model '{model_name}' is not supported. "
            f"Available models are: {', '.join(SUPPORTED_MODELS)}"
        )

    base_model, param_grid = model_params[model_name]

    print_header(f"[{model_name}] UFC GridSearchCV Training", color = 'bright_magenta')
    time.sleep(1)
    logging.info(f"[{model_name}] 🤖 Training...")

    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='accuracy',
        error_score='raise',
        verbose=3
    )
    grid_search.fit(X_train, y_train)

    time.sleep(1)

    logging.info(
        f"[{model_name}] 🔍 Best Score: {grid_search.best_score_:.4f}\n"
        f"[{model_name}] 🔍 Best Params: {grid_search.best_params_}"
    )

    return grid_search

# === Model Selection ===
def model_factory(model_name, data_train, model_params=None):
    """
    Selects and builds a model based on the specified model name and training data.

    Args:
        model_name (str): The name of the model to be selected.
        data_train (pandas.DataFrame): The training dataset, which must include a 'label' column.

    Returns:
        object: The trained model.

    Raises:
        ValueError: If the 'label' column is missing from the training data.
        ValueError: If the model_name is invalid.
    """
    if 'label' not in data_train.columns:
        raise ValueError("The dataframe must contain a 'label' column.")

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Invalid model '{model_name}'. "
            f"Supported models are: {', '.join(SUPPORTED_MODELS)}"
        )

    X_train = data_train.drop(columns=['label'])
    y_train = data_train['label']

    return build_model(model_name, X_train, y_train, model_params)




