"""
Configuration module for the UFC Fight Predictor project.

This file centralizes all model-related configuration for the machine learning pipeline, 
including:

- Definitions of supported scikit-learn classifiers used for fight outcome prediction.
- Default hyperparameter grids for each model, structured for compatibility with GridSearchCV.
- Dictionaries mapping model classes and identifiers to human-readable names ("pretty names").
- Standardized filenames for saving/loading trained models.
- ANSI color codes for enhanced terminal output styling.

Key structures:

- default_params: 
    Dictionary mapping display names to tuples of (estimator, hyperparameter grid). 
    Used for automatic selection and hyperparameter optimization of models. 
    Each grid is tuned for balanced search space coverage and computational feasibility.

- pretty_names: 
    Maps scikit-learn class names to display-friendly names for reporting and visualization.

- pretty_model_name: 
    Maps internal model file identifiers (e.g., "rf_best") to display names, ensuring consistency 
    in file naming and result presentation.

- file_model_name: 
    Maps display names back to standardized file identifiers for model persistence.

- colors: 
    ANSI color codes for consistent and readable CLI output across scripts.

This design enforces a single source of truth for model-related settings, reducing duplication 
and maintenance overhead throughout the codebase.

Any additions or changes to model support should be made in this file to ensure consistent 
integration with the rest of the project.

Author: [Maximiliano Lioi, MSc. Mathematics]
"""
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

# Default parameters for GridSearchCV for each model
default_params = {
    "Support Vector Machine": (
        SVC(probability=True),
        {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto']}
    ),
    "Random Forest": (
        RandomForestClassifier(),
        {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 10]}
    ),
    "Logistic Regression": (
        LogisticRegression(),
        {'C': [0.01, 0.1, 1], 'solver': ['liblinear', 'lbfgs']}
    ),
    "K-Nearest Neighbors": (
        KNeighborsClassifier(),
        {'n_neighbors': [3, 7], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
    ),
    "AdaBoost": (
        AdaBoostClassifier(),
        {'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 1.0, 10.0]}
    ),
    "Naive Bayes": (
        GaussianNB(),
        {'var_smoothing': [1e-8, 1e-7, 1e-6, 1e-5]}
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(),
        {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 1.0], 'max_depth': [3, 5]}
    ),
    "Extra Trees": (
        ExtraTreesClassifier(),
        {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
    ),
    "Quadratic Discriminant Analysis": (
        QuadraticDiscriminantAnalysis(),
        {'reg_param': [0.0, 0.01, 0.1]}
    ),
    "Neural Network": (
        MLPClassifier(max_iter=200, random_state=42),
        {
            # Architecture: number and size of hidden layers
            'hidden_layer_sizes': [
                (50,), (100,), (50, 50), (100, 50)
            ],
            # Activation function for hidden layers
            'activation': ['relu', 'tanh', 'logistic'],
            # Optimizer for gradient descent
            'solver': ['adam', 'sgd'],
            # L2 regularization strength (higher alpha reduces overfitting but can cause underfitting)
            'alpha': [0.0001, 0.001, 0.01],
            # Learning rate schedule
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            # Initial learning rate (for adam and sgd solvers)
            'learning_rate_init': [0.001, 0.01],
            # Early stopping based on validation performance to prevent overfitting
            'early_stopping': [True, False],
            # Mini-batch size for training with adam/sgd
            'batch_size': [32, 64, 128],
            # Momentum for SGD (controls contribution of previous updates)
            'momentum': [0.8, 0.9],
            # Proportion of training set used as validation for early stopping
            'validation_fraction': [0.1, 0.15]
        }
    ),
    "XGBoost": (
        XGBClassifier(eval_metric='logloss'),
        {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    )
}

pretty_names = {
    "LogisticRegression": "Logistic Regression",
    "RandomForestClassifier": "Random Forest",
    "SVC": "Support Vector Machine",
    "KNeighborsClassifier": "K-Nearest Neighbors",
    "AdaBoostClassifier": "AdaBoost",
    "GaussianNB": "Naive Bayes",
    "ExtraTreesClassifier": "Extra Trees",
    "GradientBoostingClassifier": "Gradient Boosting",
    "QuadraticDiscriminantAnalysis": "Quadratic Discriminant Analysis",
    "MLPClassifier": "Neural Network",
    "XGBClassifier": "XGBoost"
}

pretty_model_name = {
    "lr_best": "Logistic Regression",
    "lr_best_no_odds": "Logistic Regression",
    "rf_best": "Random Forest",
    "rf_best_no_odds": "Random Forest",
    "svm_best": "Support Vector Machine",
    "svm_best_no_odds": "Support Vector Machine",
    "knn_best": "K-Nearest Neighbors",
    "knn_best_no_odds": "K-Nearest Neighbors",
    "ab_best": "AdaBoost",
    "ab_best_no_odds": "AdaBoost",
    "nb_best": "Naive Bayes",
    "nb_best_no_odds": "Naive Bayes",
    "et_best": "Extra Trees",
    "et_best_no_odds": "Extra Trees",
    "gb_best": "Gradient Boosting",
    "gb_best_no_odds": "Gradient Boosting",
    "qda_best": "Quadratic Discriminant Analysis",
    "qda_best_no_odds": "Quadratic Discriminant Analysis",
    "nn_best": "Neural Network",
    "nn_best_no_odds": "Neural Network",
    "xgb_best": "XGBoost",
    "xgb_best_no_odds": "XGBoost"
}

file_model_name = {
    "Logistic Regression": "lr_best",
    "Random Forest": "rf_best",
    "Support Vector Machine": "svm_best",
    "K-Nearest Neighbors": "knn_best",
    "AdaBoost": "ab_best",
    "Naive Bayes": "nb_best",
    "Extra Trees": "et_best",
    "Gradient Boosting": "gb_best",
    "Quadratic Discriminant Analysis": "qda_best",
    "Neural Network": "nn_best",
    "XGBoost": "xgb_best"
}

# Extended ANSI color codes for console outputs
colors = {
    "default": "\033[0m",
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "gray": "\033[90m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m",
    "bright_cyan": "\033[96m",
    "bright_white": "\033[97m",
    "bold": "\033[1m",
    "underline": "\033[4m",
    "reverse": "\033[7m"
}
