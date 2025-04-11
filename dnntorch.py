import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Optional


class EarlyStopping:
    """
    EarlyStopping monitors a metric (e.g., loss) and stops training
    if no improvement is observed for a specified 'patience' number of epochs.

    - If 'report_minimize' is used, it assumes a lower score is better (e.g., MSE).
    - If 'report_maximize' is used, it assumes a higher score is better (e.g., accuracy).
    """

    def __init__(self, patience: int = 100, error_difference_needed: float = 1e-4):
        """
        Args:
            patience: Number of epochs to wait before stopping if no improvement.
            error_difference_needed: Minimum delta change required to qualify as an improvement.
        """
        self.patience = patience
        self.best_score = None
        self.error_difference_needed = error_difference_needed
        self.best_epoch = None

    def report_minimize(self, score: float, epoch: int) -> Tuple[int, bool]:
        """
        Use this when a lower 'score' is better (e.g., MSE).

        Args:
            score: The current metric value for the epoch.
            epoch: The current epoch number.

        Returns:
            best_epoch: The best (lowest) metric epoch so far.
            stop: Whether training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif (self.best_score - score) > self.error_difference_needed:
            self.best_score = score
            self.best_epoch = epoch

        stop = (epoch - self.best_epoch) >= self.patience
        return self.best_epoch, stop

    def report_maximize(self, score: float, epoch: int) -> Tuple[int, bool]:
        """
        Use this when a higher 'score' is better (e.g., accuracy, R^2).

        Args:
            score: The current metric value for the epoch.
            epoch: The current epoch number.

        Returns:
            best_epoch: The best (highest) metric epoch so far.
            stop: Whether training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif (score - self.best_score) > self.error_difference_needed:
            self.best_score = score
            self.best_epoch = epoch

        stop = (epoch - self.best_epoch) >= self.patience
        return self.best_epoch, stop


class Net(nn.Module):
    """
    A fully connected network (Multilayer Perceptron) with configurable:
      - number of hidden layers
      - neurons per layer
      - activation function
      - dropout
      - batch normalization
    """

    def __init__(
        self,
        num_hidden_layers: int,
        list_num_neurons: List[int],
        activation_function: str,
        dropout: float,
        in_features: int,
        out_features: int,
        seed: int = 42
    ):
        """
        Args:
            num_hidden_layers: Number of hidden layers.
            list_num_neurons: List containing the number of neurons in each hidden layer.
            activation_function: Activation function name ('relu', 'sigmoid', 'tanh', 'leaky_relu').
            dropout: Dropout probability.
            in_features: Dimensionality of input features.
            out_features: Dimensionality of output.
            seed: Random seed for reproducibility.
        """
        torch.manual_seed(seed)
        super().__init__()

        self.num_hidden_layers = num_hidden_layers
        self.list_num_neurons = list_num_neurons
        self.activation_function = self._get_activation(activation_function)
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.seed = seed

        # Build the layers (Linear + Activation + BatchNorm + Dropout)
        self.layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            if i == 0:
                self.layers.append(nn.Linear(in_features, list_num_neurons[i]))
            else:
                self.layers.append(nn.Linear(list_num_neurons[i - 1], list_num_neurons[i]))

            self.layers.append(self.activation_function)
            self.layers.append(nn.BatchNorm1d(list_num_neurons[i]))
            self.layers.append(nn.Dropout(dropout))

        # Final output layer
        self.output_layer = nn.Linear(list_num_neurons[-1], out_features)

        # Default device: GPU if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _get_activation(self, name: str) -> nn.Module:
        """
        Map the string name to an actual PyTorch activation function.
        """
        if name == "relu":
            return nn.ReLU()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "leaky_relu":
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Invalid activation function '{name}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the network.
        """
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

    def train_(
        self,
        optimizer_learning_rate: float,
        optimizer_weight_decay: float,
        early_stopper_patience: int,
        early_stopper_min_delta: float,
        max_epoch: int,
        mini_batch_size: int,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        optimizer: str,
    ) -> None:
        """
        Train the model with early stopping on the training set.

        Args:
            optimizer_learning_rate: Learning rate for the optimizer.
            optimizer_weight_decay: Weight decay (L2 regularization).
            early_stopper_patience: Patience epochs for early stopping.
            early_stopper_min_delta: Minimum delta change to qualify as improvement.
            max_epoch: Maximum number of epochs to train.
            mini_batch_size: Batch size for training; if <= 0, uses full batch gradient descent.
            X_train: Training features (torch.Tensor).
            y_train: Training targets (torch.Tensor).
            optimizer: Choice of optimizer ('ADAM' or 'SGD').
        """
        early_stop = EarlyStopping(early_stopper_patience, early_stopper_min_delta)
        loss_fn = nn.MSELoss()

        # Select optimizer
        if optimizer.upper() == "ADAM":
            optimizer = optim.Adam(
                self.parameters(),
                lr=optimizer_learning_rate,
                weight_decay=optimizer_weight_decay
            )
        elif optimizer.upper() == "SGD":
            optimizer = optim.SGD(
                self.parameters(),
                lr=optimizer_learning_rate,
                momentum=0.8,
                weight_decay=optimizer_weight_decay
            )
        else:
            raise ValueError("Invalid optimizer. Choose between 'ADAM' or 'SGD'.")

        # Move data to the selected device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        # Train loop
        if mini_batch_size > 0:
            # Mini-batch mode
            train_data_set = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_data_set, batch_size=mini_batch_size, shuffle=True)

            for epoch in range(max_epoch):
                self.train()
                for x_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    y_pred = self.forward(x_batch)
                    loss = loss_fn(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()

                self.eval()
                with torch.no_grad():
                    y_pred_train = self.forward(X_train)
                    train_loss = loss_fn(y_pred_train, y_train)
                    best_epoch, stop = early_stop.report_minimize(train_loss.item(), epoch)

                if stop:
                    print(f"Early stopping at epoch {epoch}, best epoch was {best_epoch}")
                    break

        else:
            # Full batch mode
            for epoch in range(max_epoch):
                self.train()
                optimizer.zero_grad()
                y_pred = self.forward(X_train)
                loss = loss_fn(y_pred, y_train)
                loss.backward()
                optimizer.step()

                # Check early stopping
                self.eval()
                with torch.no_grad():
                    y_pred_train = self.forward(X_train)
                    train_loss = loss_fn(y_pred_train, y_train)
                    best_epoch, stop = early_stop.report_minimize(train_loss.item(), epoch)

                if stop:
                    print(f"Early stopping at epoch {epoch}, best epoch was {best_epoch}")
                    break

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Generate predictions with the trained model.

        Args:
            X: A torch.Tensor of input features.

        Returns:
            A NumPy array of model predictions.
        """
        self.eval()
        X = X.to(self.device)
        with torch.no_grad():
            predictions = self.forward(X)
        return predictions.cpu().numpy()

    def save(self, path: str) -> None:
        """
        Save the model's state_dict and key hyperparameters.

        Args:
            path: File path for saving the model checkpoint.
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_hidden_layers': self.num_hidden_layers,
            'list_num_neurons': self.list_num_neurons,
            'activation_function': self.activation_function,
            'dropout': self.dropout,
            'in_features': self.in_features,
            'out_features': self.out_features,
            'seed': self.seed
        }, path)

    @staticmethod
    def load(path: str, activation_function: Optional[str] = None, dropout: Optional[float] = None) -> "Net":
        """
        Load a saved model from a checkpoint file.

        Args:
            path: Path of the saved model.
            activation_function: If provided, overrides the saved activation function.
            dropout: If provided, overrides the saved dropout value.

        Returns:
            An instance of Net with weights loaded.
        """
        checkpoint = torch.load(path)
        # If user manually specifies activation or dropout, override
        if activation_function is not None:
            checkpoint['activation_function'] = activation_function
        else:
            # The activation function stored is an actual PyTorch activation object
            # so we try to get its class name if possible
            if hasattr(checkpoint['activation_function'], '__class__'):
                # e.g., nn.ReLU -> "relu"
                checkpoint['activation_function'] = checkpoint['activation_function'].__class__.__name__.lower()
            elif isinstance(checkpoint['activation_function'], str):
                # Already a string
                pass
            else:
                raise ValueError("Activation function in checkpoint is not recognized.")

        if dropout is not None:
            checkpoint['dropout'] = dropout

        model = Net(
            num_hidden_layers=checkpoint['num_hidden_layers'],
            list_num_neurons=checkpoint['list_num_neurons'],
            activation_function=checkpoint['activation_function'],
            dropout=checkpoint['dropout'],
            in_features=checkpoint['in_features'],
            out_features=checkpoint['out_features'],
            seed=checkpoint.get('seed', 42)
        )

        # Load the model state
        model.load_state_dict(checkpoint['model_state_dict'])

        return model
