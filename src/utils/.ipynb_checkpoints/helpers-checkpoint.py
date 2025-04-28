import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.config import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

def split_and_standardize(data: pd.DataFrame, categorical_columns: list, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the input DataFrame into training and testing sets, and standardizes
    only the numerical feature columns (deduced from excluding categorical_columns and 'label').

    Args:
        data (pd.DataFrame): Input DataFrame with features and a 'label' column.
        categorical_columns (list): List of column names considered categorical (not to be standardized).
        test_size (float, optional): Proportion of the dataset to include in the test split.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        tuple:
            - pd.DataFrame: data_train with standardized numerical features and original labels.
            - pd.DataFrame: data_test with standardized numerical features and original labels.
    """
    if data.empty:
        raise ValueError("Input DataFrame is empty.")
    if 'label' not in data.columns:
        raise ValueError("Input DataFrame must contain a 'label' column.")
    if data.isnull().sum().any():
        raise ValueError("DataFrame contains missing values. Please clean the data before proceeding.")

    # Deduce numerical columns (excluding 'label' and categorical columns)
    excluded_columns = set(categorical_columns + ['label'])
    numerical_columns = [col for col in data.columns if col not in excluded_columns]

    # Separate features and label
    X = data.drop(columns='label')
    y = data['label']

    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standardize only numerical columns
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])

    # Combine scaled features with target
    data_train = pd.concat([X_train_scaled, y_train], axis=1).reset_index(drop=True)
    data_test = pd.concat([X_test_scaled, y_test], axis=1).reset_index(drop=True)

    print_header('Numerical Data has been standardized and the dataset has been split', color = 'bright_green')
    return data_train, data_test

    
def get_predictions(model, X_test):
    """
    Returns model predictions and probabilities based on the input model type.
    
    Args:
        model (object): The trained model (either PyTorch or scikit-learn).
        X_test (numpy.ndarray): The input features for testing.

    Returns:
        tuple: A tuple containing the predictions and probabilities.
    """
    # For scikit-learn models, check if the model has predict_proba or decision_function
    if hasattr(model, "predict_proba"):
        # For models with a predict_proba method (like RandomForest, AdaBoost, etc.)
        probs = model.predict_proba(X_test)[:, 1]
    else:
        # For models like SVM or Logistic Regression, use decision_function
        probs = model.decision_function(X_test)
        # Apply sigmoid for SVMs or linear models to convert to probabilities
        probs = 1 / (1 + np.exp(-probs))
    
    preds = model.predict(X_test)
    return preds, probs

def parameters_dict(models_dict):
    # Diccionario para almacenar los modelos y sus mejores parámetros
    best_params = {}
    
    # Iterar sobre los modelos en models_dict
    for model_name, model in models_dict.items():
        # Verificar si el modelo tiene un atributo 'best_params_'
        if hasattr(model, 'best_params_'):
            best_params[model_name] = model.best_params_
        else:
            # Si no tiene 'best_params_', se coloca un mensaje de error
            best_params[model_name] = 'No GridSearchCV parameters'
    return best_params
    
def print_header(text: str, color: str = "default", padding_side = 2, padding_top_bottom = 0) -> None:
    """
    Prints a beautified and centered string inside a stylish ASCII box, with optional color.

    Example:
    >>> print_header("Training UFC Fight Predictor Model", color="cyan")
    """
    color_code = colors.get(color.lower(), colors["default"])
    text_line = f"{text.center(len(text) + padding_side * 2)}"
    width = len(text_line)

    top_border = f"╔{'═' * width}╗"
    empty_line = f"║{' ' * width}║"
    middle_line = f"║{text_line}║"
    bottom_border = f"╚{'═' * width}╝"

    lines = [top_border]
    lines.extend([empty_line] * padding_top_bottom)
    lines.append(middle_line)
    lines.extend([empty_line] * padding_top_bottom)
    lines.append(bottom_border)

    # Print with color
    print(color_code + "\n".join(lines) + colors["default"])

def get_pretty_model_name(model) -> str:
    """
    Returns the pretty name of the model type.
    If the model is wrapped in a GridSearchCV, it retrieves the base model.

    Example:
    >>> get_pretty_model_name(knn_model)
    'K-Nearest Neighbors'
    """
    
    base_model = model.best_estimator_
    model_name = type(base_model).__name__

    # Si el modelo no tiene un nombre bonito mapeado, lanzamos un error
    if model_name not in pretty_names:
        raise ValueError(f"Model '{model_name}' does not have a predefined pretty name in the mapping.")

    # Devuelve el nombre bonito
    return pretty_names[model_name]

def get_supported_models():
    """
    Returns a sorted list of all supported model names.

    Returns:
        list: List of model names (str) available in default_params.
    """
    return sorted(default_params.keys())