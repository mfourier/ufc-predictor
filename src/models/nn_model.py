# import torch
# import torch.nn as nn
# import torch.optim as optim
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