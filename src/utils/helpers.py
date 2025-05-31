import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .print_utils import print_header 

def split_and_standardize(
    data: pd.DataFrame,
    categorical_columns: list[str],
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the input DataFrame into train/test sets and standardize only numerical features.

    Args:
        data (pd.DataFrame): DataFrame with feature columns and a 'label' column.
        categorical_columns (list[str]): Column names that should not be standardized.
        test_size (float): Proportion of test set.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (data_train, data_test) with standardized numerical features.
    """
    if data.empty:
        raise ValueError("Input DataFrame is empty.")
    if 'label' not in data.columns:
        raise ValueError("Input DataFrame must contain a 'label' column.")
    if data.isnull().sum().any():
        raise ValueError("DataFrame contains missing values. Please clean the data before proceeding.")

    excluded_columns = set(categorical_columns + ['label'])
    numerical_columns = [col for col in data.columns if col not in excluded_columns]

    X = data.drop(columns='label')
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])

    data_train = pd.concat([X_train_scaled, y_train], axis=1).reset_index(drop=True)
    data_test = pd.concat([X_test_scaled, y_test], axis=1).reset_index(drop=True)

    print_header("Numerical data standardized and split complete", color="bright_green")
    return data_train, data_test


def get_predictions(model, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions and probabilities using the input model.

    Args:
        model (object): Trained scikit-learn or compatible model.
        X_test (np.ndarray): Test feature matrix.

    Returns:
        tuple: (predictions, probabilities)
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.decision_function(X_test)
        probs = 1 / (1 + np.exp(-probs))  # sigmoid

    preds = model.predict(X_test)
    return preds, probs


def parameters_dict(models_dict: dict) -> dict:
    """
    Extract the best parameters from a dictionary of trained models.

    Args:
        models_dict (dict): Dictionary of models keyed by name.

    Returns:
        dict: Dictionary of best parameters or a default message if unavailable.
    """
    best_params = {}
    for model_name, model in models_dict.items():
        if hasattr(model, 'best_params_'):
            best_params[model_name] = model.best_params_
        else:
            best_params[model_name] = 'No GridSearchCV parameters'
    return best_params
