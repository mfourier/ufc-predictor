import pandas as pd
import numpy as np
from typing import Optional, Sequence
from sklearn.base import BaseEstimator

from .metrics import compute_metrics, plot_confusion_matrix, DEFAULT_METRICS
from .helpers import get_pretty_model_name, get_predictions, print_header
import logging

logger = logging.getLogger(__name__)

class UFCPredictorClass:
    """
    Wrapper for UFC model prediction, evaluation, and reporting.

    Attributes:
        model (BaseEstimator): Trained scikit-learn model (or GridSearchCV).
        data_train (pd.DataFrame): Training data.
        data_test (pd.DataFrame): Testing data.
        name (str): Human-readable name of the model.
        best_params (dict): Best parameters if GridSearchCV was used.
        last_results (dict): Last computed evaluation metrics.
    """

    def __init__(
        self,
        model: BaseEstimator
    ):
        """
        Initialize the UFC Predictor with a model and its corresponding datasets.

        Args:
            model (BaseEstimator): Trained model or GridSearchCV object.
        """
        self.model = model
        self.name = get_pretty_model_name(model)
        self.hyperparameters = getattr(model, "best_params_", None)
        self.score = getattr(model, "best_score_", None)
        self.statistics: Optional[dict[str, float]] = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the internal model.

        Args:
            X (pd.DataFrame): Feature set.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate prediction probabilities using the internal model.

        Args:
            X (pd.DataFrame): Feature set.

        Returns:
            np.ndarray: Class probabilities.

        Raises:
            AttributeError: If the model does not support probability prediction.
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError("This model does not support predict_proba.")

    def summary(self) -> None:
        """
        Print a summary of the model, its parameters, and latest evaluation results.
        """
        print_header(f"Model: {self.name}", color='bright_blue')
        if self.best_params:
            print(f"Best Params: {self.best_params}")
        if self.last_results:
            print("Last evaluation results:")
            for k, v in self.last_results.items():
                print(f"{k:>12}: {v:.4f}")
