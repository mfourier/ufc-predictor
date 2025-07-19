import logging
import time
from typing import Optional
import pandas as pd
from sklearn.model_selection import GridSearchCV
from src.helpers import *
from src.data import UFCData
from src.config import *

# Configure logging
logging.basicConfig(level=logging.INFO)

SUPPORTED_MODELS: list[str] = list(default_params.keys())

def model_factory(
        model_name: str,
        UFCData: UFCData,
        model_params: Optional[dict] = None,
        scoring: str = 'accuracy'
    ) -> GridSearchCV:
    """
    Selects and builds a model based on the specified model name and a UFCData object.

    Args:
        model_name (str): Identifier of the model to build (must be in SUPPORTED_MODELS).
        UFCData (UFCData): An instance of UFCData containing preprocessed and standardized training data.
        model_params (dict, optional): Dictionary with model instances and hyperparameters.
        scoring (str, optional): Metric to use for hyperparameter tuning. Default is 'accuracy'.

    Returns:
        GridSearchCV: A trained GridSearchCV object with the best estimator.

    Raises:
        ValueError: If model name is invalid or processed training data is not available.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Invalid model '{model_name}'. "
            f"Supported models are: {', '.join(SUPPORTED_MODELS)}"
        )

    try:
        X_train, y_train = UFCData.get_processed_train()
    except ValueError as e:
        raise ValueError(f"Error retrieving standardized training data: {e}")

    return build_model(model_name, X_train, y_train, model_params, scoring)


def build_model(
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_params: Optional[dict] = None,
        scoring: str = 'accuracy'
    ) -> GridSearchCV:
    """
    Constructs and trains a model using GridSearchCV based on the given model name.

    Args:
        model_name (str): Name of the model to build and train (e.g., 'svm', 'random_forest').
        X_train (pd.DataFrame): Training input features.
        y_train (pd.Series): Training target labels.
        model_params (dict, optional): Dictionary specifying the base model and hyperparameter grid.
        scoring (str, optional): Metric used to evaluate models during grid search.

    Returns:
        GridSearchCV: A trained GridSearchCV object containing the best estimator.

    Raises:
        ValueError: If the specified model name is not in the supported list.
    """
    if model_params is None:
        model_params = default_params

    if model_name not in model_params:
        raise ValueError(
            f"Model '{model_name}' is not supported. "
            f"Available models are: {', '.join(SUPPORTED_MODELS)}"
        )

    base_model, param_grid = model_params[model_name]

    logger.info(f"[{model_name}] UFC GridSearchCV Training ({scoring.capitalize()}))

    time.sleep(1)
    logging.info(f"[{model_name}] 🤖 Training...")

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring=scoring,
        error_score='raise',
        verbose=3
    )
    grid_search.fit(X_train, y_train)

    time.sleep(1)
    logging.info(
        f"[{model_name}] 🔍 Best {scoring.capitalize()}: {grid_search.best_score_:.4f}\n"
        f"[{model_name}] 🔍 Best Params: {grid_search.best_params_}"
    )

    return grid_search
