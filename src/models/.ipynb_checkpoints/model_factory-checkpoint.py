import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

class NeuralNetworkModel(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes=(100,), output_dim=1):
        super(NeuralNetworkModel, self).__init__()
        layers = []
        prev_size = input_dim
        for size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def fit(self, X, y, epochs=100, batch_size=32, lr=0.001):
        self.train()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        criterion = nn.BCEWithLogitsLoss()  # Usamos esta función de pérdida para clasificación binaria
        optimizer = optim.Adam(self.parameters(), lr=lr)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs.view(-1), batch_y)
                loss.backward()
                optimizer.step()
        
        return self

    def predict(self, X):
        self.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self(X_tensor)
        return (outputs.view(-1) > 0).float()  # Devuelve 0 o 1 para clasificación binaria

def build_neural_network(X_train, y_train):
    input_dim = X_train.shape[1]
    model = NeuralNetworkModel(input_dim=input_dim, hidden_layer_sizes=(100,))

    # Aquí puedes ajustar más parámetros de entrenamiento como epochs, batch_size, etc.
    model.fit(X_train, y_train, epochs=100, batch_size=32, lr=0.001)
    
    return model

def build_svm(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto'],
    }
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    
    # 🔹 Best Parameters from Grid Search
    print(f"Best Parameters from GridSearchCV: {grid_search.best_params_}")
    
    return grid_search.best_estimator_

def build_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20],
    }
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)

    # 🔹 Best Parameters from Grid Search
    print(f"Best Parameters from GridSearchCV: {grid_search.best_params_}")

    return grid_search.best_estimator_

def build_logistic_regression(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs'],
    }
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)

    # 🔹 Best Parameters from Grid Search
    print(f"Best Parameters from GridSearchCV: {grid_search.best_params_}")

    return grid_search.best_estimator_

def build_knn(X_train, y_train):
    param_grid = {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    }
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)

    # 🔹 Best Parameters from Grid Search
    print(f"Best Parameters from GridSearchCV: {grid_search.best_params_}")
    
    return grid_search.best_estimator_

def build_adaboost(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0],
    }
    grid_search = GridSearchCV(AdaBoostClassifier(), param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters from GridSearchCV (AdaBoost): {grid_search.best_params_}")
    return grid_search.best_estimator_

def build_naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def get_model(model_name, X_train, y_train):
    if model_name == "svm":
        return build_svm(X_train, y_train)
    elif model_name == "random_forest":
        return build_random_forest(X_train, y_train)
    elif model_name == "logistic_regression":
        return build_logistic_regression(X_train, y_train)
    elif model_name == "knn":
        return build_knn(X_train, y_train)
    elif model_name == "neural_network":
        return build_neural_network(X_train, y_train)
    elif model_name == "adaboost":
        return build_adaboost(X_train, y_train)
    elif model_name == "naive_bayes":
        return build_naive_bayes(X_train, y_train)
    else:
        raise ValueError(f"Modelo '{model_name}' no soportado.")

