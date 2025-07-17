import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional, Sequence
from sklearn.base import BaseEstimator
from sklearn.metrics import ConfusionMatrixDisplay
from src.helpers import get_pretty_model_name, print_header

class UFCModel:
    def __init__(self, model: BaseEstimator):
        """
        Initialize the UFCModel wrapper.

        Parameters
        ----------
        model : BaseEstimator
            A trained scikit-learn model or GridSearchCV object.
        """
        self.model = model
        self.name = get_pretty_model_name(model)
        self.best_params_ = getattr(model, "best_params_", None)
        self.score = getattr(model, "best_score_", None)
        self.metrics: Optional[dict[str, float]] = None
        self.cm = None

    @property
    def estimator(self):
        """
        Get the final estimator.

        Returns
        -------
        BaseEstimator
            The best estimator if GridSearchCV, else the model itself.
        """
        return getattr(self.model, 'best_estimator_', self.model)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        return self.estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Probability estimates.

        Raises
        ------
        AttributeError
            If the estimator does not support predict_proba.
        """
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)
        raise AttributeError(f"Model '{self.name}' does not support predict_proba.")

    def score_model(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Compute the accuracy score.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.
        y : pd.Series
            True labels.

        Returns
        -------
        float
            Mean accuracy score.
        """
        return self.estimator.score(X, y)

    def get_params(self, deep: bool = True) -> dict:
        """
        Get estimator parameters.

        Parameters
        ----------
        deep : bool, optional
            If True, return parameters of subobjects.

        Returns
        -------
        dict
            Parameter names and values.
        """
        return self.estimator.get_params(deep=deep)

    def set_params(self, **params):
        """
        Set estimator parameters.

        Parameters
        ----------
        **params
            Parameter names and values.

        Returns
        -------
        self
            Updated UFCModel instance.
        """
        self.estimator.set_params(**params)
        return self

    @property
    def classes_(self):
        """
        Get class labels.

        Returns
        -------
        np.ndarray or None
            Known class labels.
        """
        return getattr(self.estimator, "classes_", None)

    @property
    def n_features_in_(self):
        """
        Get number of input features.

        Returns
        -------
        int or None
            Number of features seen during fit.
        """
        return getattr(self.estimator, "n_features_in_", None)

    def plot_cm(self) -> None:
        """
        Plot the stored confusion matrix.

        Raises
        ------
        ValueError
            If confusion matrix is not computed.
        """
        if self.cm is None:
            raise ValueError("Confusion matrix is not available. Please compute it before plotting.")
        print_header(f"Confusion Matrix: {self.name}", color='bright_cyan')
        disp = ConfusionMatrixDisplay(confusion_matrix=self.cm)
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, feature_names: Optional[Sequence[str]] = None, max_display: int = 20) -> None:
        """
        Plot feature importances or coefficients.

        Parameters
        ----------
        feature_names : sequence of str, optional
            Names of features.
        max_display : int, optional
            Max number of top features to show.

        Raises
        ------
        AttributeError
            If the estimator does not provide feature importances.
        ValueError
            If feature_names length mismatches feature count.
        """
        model = self.estimator
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            title = f"Feature Importances - {self.name}"
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_.ravel())
            title = f"Absolute Coefficient Importances - {self.name}"
        else:
            raise AttributeError(f"Model '{self.name}' does not provide feature importances (feature_importances_ or coef_).")

        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        elif len(feature_names) != len(importances):
            raise ValueError(f"Length of feature_names ({len(feature_names)}) does not match number of features ({len(importances)}).")

        sorted_idx = np.argsort(importances)[::-1][:max_display]
        top_features = [feature_names[i] for i in sorted_idx]
        top_importances = importances[sorted_idx]
        plt.figure(figsize=(10, max(4, len(top_features) // 2)))
        plt.barh(top_features[::-1], top_importances[::-1])
        plt.xlabel("Importance")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_feature_importances_grid(models: list, feature_names: Optional[Sequence[str]] = None, max_display: int = 20, save_file: bool = False, filename: str = "model_feature_importances_grid.png") -> None:
        """
        Plot feature importances for multiple models.

        Parameters
        ----------
        models : list of UFCModel
            List of UFCModel instances.
        feature_names : sequence of str, optional
            Names of features.
        max_display : int, optional
            Max number of top features to show.
        save_file : bool, optional
            Whether to save the figure.
        filename : str, optional
            Name of the saved file.
        """
        filtered = [m for m in models if hasattr(m.estimator, "feature_importances_") or hasattr(m.estimator, "coef_")]
        if not filtered:
            print("❌ No models with feature importances or coefficients.")
            return
        n_models = len(filtered)
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
        axes = axes.flatten()
        for i, model in enumerate(filtered):
            ax = axes[i]
            mdl = model.estimator
            if hasattr(mdl, "feature_importances_"):
                importances = mdl.feature_importances_
                title = f"{model.name} (Importances)"
            else:
                importances = np.abs(mdl.coef_.ravel())
                title = f"{model.name} (Coefficients)"
            fnames = feature_names or [f"Feature {j}" for j in range(len(importances))]
            if len(fnames) != len(importances):
                raise ValueError(f"Length mismatch between feature_names and importances in model {model.name}.")
            sorted_idx = np.argsort(importances)[::-1][:max_display]
            top_features = [fnames[j] for j in sorted_idx]
            top_importances = importances[sorted_idx]
            ax.barh(top_features[::-1], top_importances[::-1])
            ax.set_title(title)
            ax.set_xlabel("Importance")
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        if save_file:
            img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../img/"))
            os.makedirs(img_dir, exist_ok=True)
            save_path = os.path.join(img_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to: {save_path}")
        plt.show()

    def __repr__(self) -> str:
        """
        String representation of the model.

        Returns
        -------
        str
            Summary string.
        """
        summary = [f"<UFCModel: {self.name}>"]
        if self.score is not None:
            summary.append(f"  - Best CV Score : {self.score:.4f}")
        if self.best_params_:
            summary.append(f"  - Best Params   : {self.best_params_}")
        if self.metrics:
            summary.append("  - Last Evaluation:")
            for k, v in self.metrics.items():
                summary.append(f"      {k:<12}: {v:.4f}")
        return "\n".join(summary)

    def summary(self) -> None:
        """
        Print a decorated summary of the model.
        """
        print_header(f"Model: {self.name}", color='bright_blue')
        print(self)
