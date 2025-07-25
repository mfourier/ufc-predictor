# Author: Maximiliano Lioi | License: MIT

import logging
from typing import Optional, Union, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
)
from src.helpers import get_predictions
from src.data import UFCData
from src.model import UFCModel

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_METRICS = [
    'Accuracy',
    'Balanced Accuracy',
    'Precision Red', 'Recall Red', 'F1 Red',
    'Precision Blue', 'Recall Blue', 'F1 Blue',
    'F1 Macro',
    'ROC AUC', 'Brier Score',
    'MCC',
    'Kappa'
]


def evaluate_metrics(
        model: object,
        UFCData: UFCData,
        verbose: bool = False,
        metrics_to_compute: Optional[Sequence[str]] = None
    ) -> dict[str, float]:
    """
    Evaluate a trained UFCModel on test data stored in a UFCData object.

    Args:
        model (UFCModel): A trained model wrapper.
        ufc_data (UFCData): Dataset handler with standardized test data.
        verbose (bool): Whether to log detailed results.
        metrics_to_compute (list, optional): Metrics to evaluate.

    Returns:
        dict[str, float]: Computed metric results.
    """
    X_test, y_test = UFCData.get_processed_test()
    metrics_to_compute = metrics_to_compute or DEFAULT_METRICS

    try:
        preds, probs = get_predictions(model, X_test)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

    results = compute_metrics(y_test, preds, probs, metrics_to_compute)

    if verbose:
        logger.info("=" * 50)
        logger.info(f"📊 Evaluation for: [{model.name}]")
        if model.best_params_:
            logger.info(f"✨ Best Parameters: {model.best_params_}")
        for k, v in results.items():
            logger.info(f"{k:>12}: {v:.4f}")
        logger.info("=" * 50)

    return results


def evaluate_cm(
        model: UFCModel,
        ufc_data: UFCData,
    ) -> np.ndarray:
    """
    Compute and store the confusion matrix for a UFCModel using UFCData.

    Args:
        model (UFCModel): A trained model.
        ufc_data (UFCData): Dataset handler with standardized test data.

    Returns:
        np.ndarray: Confusion matrix.
    """
    X_test, y_test = ufc_data.get_processed_test()
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    return cm


def compute_metrics(
        y_test: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        y_proba: Optional[Union[np.ndarray, list]],
        metrics_to_compute: Sequence[str]
) -> dict[str, float]:
    results: dict[str, float] = {}

    for metric in metrics_to_compute:
        if metric == 'Accuracy':
            results['Accuracy'] = accuracy_score(y_test, y_pred)
        elif metric == 'Balanced Accuracy':
            results['Balanced Accuracy'] = balanced_accuracy_score(y_test, y_pred)
        elif metric == 'Precision Red':
            results['Precision Red'] = precision_score(y_test, y_pred, pos_label=0, zero_division=1)
        elif metric == 'Recall Red':
            results['Recall Red'] = recall_score(y_test, y_pred, pos_label=0, zero_division=1)
        elif metric == 'F1 Red':
            results['F1 Red'] = f1_score(y_test, y_pred, pos_label=0, zero_division=1)
        elif metric == 'Precision Blue':
            results['Precision Blue'] = precision_score(y_test, y_pred, pos_label=1, zero_division=1)
        elif metric == 'Recall Blue':
            results['Recall Blue'] = recall_score(y_test, y_pred, pos_label=1, zero_division=1)
        elif metric == 'F1 Blue':
            results['F1 Blue'] = f1_score(y_test, y_pred, pos_label=1, zero_division=1)
        elif metric == 'F1 Macro':
            results['F1 Macro'] = f1_score(y_test, y_pred, average='macro', zero_division=1)
        elif metric == 'ROC AUC' and y_proba is not None:
            results['ROC AUC'] = roc_auc_score(y_test, y_proba)
        elif metric == 'Brier Score' and y_proba is not None:
            results['Brier Score'] = brier_score_loss(y_test, y_proba)
        elif metric == 'MCC':
            results['MCC'] = matthews_corrcoef(y_test, y_pred)
        elif metric == 'Kappa':
            results['Kappa'] = cohen_kappa_score(y_test, y_pred)
        else:
            logger.warning(f"Unsupported or unavailable metric: {metric}")

    return {k: round(v, 4) for k, v in results.items()}


def compare_metrics(
        models_list: list[UFCModel],
    ) -> pd.DataFrame:
    """
    Compare multiple UFCModel objects using stored metrics.

    Args:
        models_list (list): List of trained UFCModel instances.

    Returns:
        pd.DataFrame: Table comparing model performance.
    """
    logger.info("🔍 Starting comparison of models...")
    results = []

    for model in models_list:
        logger.info(f"Evaluating model: {model.name}")
        if model.metrics is None:
            logger.warning(f"Model {model.name} has no stored metrics.")
            continue
        row = model.metrics.copy()
        row['Model'] = model.name
        results.append(row)

    df = pd.DataFrame(results).set_index('Model')
    logger.info("✅ Comparison completed.")
    return df


def best_model_per_metric(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify the best-performing model per metric.

    Args:
        metrics_df (pd.DataFrame): DataFrame with models and metrics.

    Returns:
        pd.DataFrame: Best model and score for each metric.
    """
    best = []
    for metric in metrics_df.columns:
        if metric == 'Brier Score':
            best_model = metrics_df[metric].idxmin()
            best_value = metrics_df[metric].min()
        else:
            best_model = metrics_df[metric].idxmax()
            best_value = metrics_df[metric].max()

        best.append({"Metric": metric, "Best Model": best_model, "Value": round(best_value, 4)})
        logger.info(f"🏅 Best model for {metric}: {best_model} ({best_value:.4f})")

    return pd.DataFrame(best)
