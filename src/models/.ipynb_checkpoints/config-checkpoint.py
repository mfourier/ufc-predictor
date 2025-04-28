from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

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
        "QuadraticDiscriminantAnalysis": "Quadratic Discriminant Analysis"
    }

pretty_model_names = {
        "lr_best": "Logistic Regression",
        "rf_best": "Random Forest",
        "svm_best": "Support Vector Machine",
        "knn_best": "K-Nearest Neighbors",
        "ab_best": "AdaBoost",
        "nb_best": "Naive Bayes",
        "et_best": "Extra Trees",
        "gb_best": "Gradient Boosting",
        "qda_best": "Quadratic Discriminant Analysis"
    }

# ANSI color codes
# Extended ANSI color codes
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
