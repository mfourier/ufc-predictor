import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import logging

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

# Logging config
logging.basicConfig(level=logging.INFO)

# === Neural Network ===
class NeuralNetworkModel(nn.Module):
    """
    A simple feed-forward neural network model for binary classification.
    
    Attributes:
        network (nn.Sequential): A sequential container for the layers of the network.
    
    Methods:
        __init__(input_dim, hidden_layer_sizes, output_dim): Initializes the network with given parameters.
        forward(x): Defines the forward pass for the network.
        fit(X, y, epochs, batch_size, lr): Trains the model using the given data.
        predict(X): Predicts binary outcomes (0 or 1) based on input data.
        predict_proba(X): Predicts probabilities of binary outcomes (0 and 1) for the input data.
    """

    def __init__(self, input_dim, hidden_layer_sizes=(100,), output_dim=1):
        """
        Initializes the neural network with the specified architecture.
        
        Args:
            input_dim (int): The number of features in the input data.
            hidden_layer_sizes (tuple): The number of units in each hidden layer (default is (100,)).
            output_dim (int): The number of output units (default is 1 for binary classification).
        """
        super().__init__()
        layers = []
        prev_size = input_dim
        for size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the forward pass for the network.
        
        Args:
            x (torch.Tensor): The input tensor to the network.
        
        Returns:
            torch.Tensor: The output tensor of the network.
        """
        return self.network(x)

    def fit(self, X, y, epochs=100, batch_size=32, lr=0.001):
        """
        Trains the neural network on the provided training data.
        
        Args:
            X (numpy.ndarray): The input features for training.
            y (numpy.ndarray): The target labels for training.
            epochs (int): The number of training epochs (default is 100).
            batch_size (int): The batch size for training (default is 32).
            lr (float): The learning rate for training (default is 0.001).
        
        Returns:
            self: The trained model.
        """
        self.train()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs.view(-1), batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        """
        Predicts binary outcomes (0 or 1) based on input features.
        
        Args:
            X (numpy.ndarray): The input features for prediction.
        
        Returns:
            numpy.ndarray: The predicted binary outcomes (0 or 1).
        """
        self.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self(X_tensor)
        return ((outputs.view(-1) > 0).int()).numpy()

    def predict_proba(self, X):
        """
        Predicts probabilities for binary outcomes (0 and 1).
        
        Args:
            X (numpy.ndarray): The input features for prediction.
        
        Returns:
            numpy.ndarray: The predicted probabilities for each class (0 and 1).
        """
        self.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self(X_tensor)
        probs = torch.sigmoid(outputs.view(-1))
        return np.stack([1 - probs, probs], axis=1)  

# === GridSearch Model Constructor===
def build_model(model_name, X_train, y_train):
    """
    Constructs and trains a model based on the specified model name.
    
    Args:
        model_name (str): The name of the model to build and train (e.g., 'neural_network', 'svm').
        X_train (numpy.ndarray): The training input features.
        y_train (numpy.ndarray): The training target labels.
    
    Returns:
        object: The trained model.
    
    Raises:
        ValueError: If the specified model name is not supported.
    """
    model_name = model_name.lower()

    if model_name == "neural_network":
        input_dim = X_train.shape[1]
        model = NeuralNetworkModel(input_dim=input_dim, hidden_layer_sizes=(100,))
        model.fit(X_train, y_train, epochs=100, batch_size=32, lr=0.001)
        return model

    # GridSearch Models Dictionary 
    model_params = {
        "svm": (
            SVC(probability=True),
            {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto']}
        ),
        "random_forest": (
            RandomForestClassifier(),
            {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]}
        ),
        "logistic_regression": (
            LogisticRegression(),
            {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
        ),
        "knn": (
            KNeighborsClassifier(),
            {'n_neighbors': [3, 5, 7, 10], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
        ),
        "adaboost": (
            AdaBoostClassifier(),
            {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}
        ),
        "naive_bayes": (
            GaussianNB(),
            {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}
        )
    }

    if model_name not in model_params:
        raise ValueError(f"Model '{model_name}' is not supported.")

    base_model, param_grid = model_params[model_name]

    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='f1', error_score='raise')
    grid_search.fit(X_train, y_train)

    logging.info(f"[{model_name.upper()}] Best F1: {grid_search.best_score_:.4f}")
    logging.info(f"[{model_name.upper()}] Best Params: {grid_search.best_params_}")

    return grid_search.best_estimator_

# === Model Selection ===
def get_model(model_name, data_train):
    """
    Selects and builds a model based on the specified model name and training data.
    
    Args:
        model_name (str): The name of the model to be selected.
        data_train (pandas.DataFrame): The training dataset, which must include a 'label' column.
    
    Returns:
        object: The trained model.
    
    Raises:
        ValueError: If the 'label' column is missing from the training data.
    """
    if 'label' not in data_train.columns:
        raise ValueError("The dataframe must contain a 'label' column.")

    X_train = data_train.drop(columns=['label']).values
    y_train = data_train['label'].values

    return build_model(model_name, X_train, y_train)




