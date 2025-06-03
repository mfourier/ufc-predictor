import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.config import colors, pretty_names, default_params
from datetime import datetime

def get_predictions(model: object, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions and probabilities using the input model.

    Args:
        model (UFCModel): A trained UFCModel.
        X_test (np.ndarray): Test feature matrix.

    Returns:
        tuple: (predictions, probabilities)
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError("Model doesn't have a 'predict_proba' atributte.")
    preds = model.predict(X_test)
    return preds, probs

def print_header(
    text: str,
    color: str = "default",
    padding_side: int = 2,
    padding_top_bottom: int = 0
) -> None:
    """
    Print a centered message in an ASCII-styled box with optional color formatting.

    Args:
        text (str): The message to print.
        color (str): Color name defined in the colors dictionary.
        padding_side (int): Horizontal padding on each side.
        padding_top_bottom (int): Number of empty lines above and below the text.
    
    Example:
        >>> print_header("Training started", color="cyan")
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

    print(color_code + "\n".join(lines) + colors["default"])

def get_pretty_model_name(model: object) -> str:
    """
    Return the display-friendly name of a trained model.

    If the model is a GridSearchCV wrapper, extract the base estimator.

    Args:
        model (UFCModel): A trained UFCModel.

    Returns:
        str: Human-readable model name defined in `pretty_names`.

    Raises:
        ValueError: If the model's class is not mapped in `pretty_names`.
    """
    base_model = model.best_estimator_
    model_name = type(base_model).__name__

    if model_name not in pretty_names:
        raise ValueError(
            f"Model '{model_name}' does not have a predefined pretty name in the mapping."
        )

    return pretty_names[model_name]


def get_supported_models() -> list[str]:
    """
    Retrieve all supported model identifiers defined in `default_params`.

    Returns:
        list[str]: Sorted list of model names.
    """
    return sorted(default_params.keys())

def log_training_result(
    model_name: str,
    best_params: dict,
    metrics: dict,
    duration: float,
    log_path: str = "../data/results/training_log.csv"
) -> None:
    """
    Log the training results of a model into a cumulative CSV file.

    Args:
        model_name (str): Name of the model used in training.
        best_params (dict): Dictionary of hyperparameters found by GridSearchCV.
        metrics (dict): Dictionary containing evaluation metrics (accuracy, F1, etc.).
        duration (float): Duration of training in seconds.
        log_path (str): Path where the CSV log will be stored.

    Returns:
        None
    """
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': model_name,
        'duration_sec': round(duration, 2),
        **metrics,
        **{f'param_{k}': v for k, v in best_params.items()}
    }

    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])

    df.to_csv(log_path, index=False)
    print(f"✅ Training logged to {log_path}")
