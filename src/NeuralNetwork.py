import warnings
from typing import List, Union, Tuple, Optional, Iterator

import numpy as np
import seaborn as sns
import sklearn.utils
from sklearn.preprocessing import StandardScaler

from src.Layer import Layer
from src.Optimizer import SGD

warnings.filterwarnings('ignore')
sns.set()


def _create_mini_batches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, drop_last: bool) -> \
        Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Creates mini-batches from the input data.

    :param X: The input features
    :param y: The target labels
    :param batch_size: The size of the mini-batches
    :param shuffle: If True, the data is shuffled before creating mini-batches
    :param drop_last: If True, the last batch is dropped if its size is smaller than batch_size
    :yield: Mini-batches (X_batch, y_batch)
    """
    if shuffle:
        X, y = sklearn.utils.shuffle(X, y)
    if drop_last:
        n_minibatches = X.shape[0] // batch_size
    else:
        n_minibatches = (X.shape[0] + batch_size - 1) // batch_size
    for i in range(n_minibatches):
        yield X[i * batch_size: (i + 1) * batch_size], y[i * batch_size: (i + 1) * batch_size]


class NeuralNetwork:
    def __init__(self, num_epochs=100, batch_size=32, optimizer=SGD(learning_rate=0.01),
                 activation='relu', hidden_sizes: List[int] = [128], num_classes=4, verbose=False, random_seed=42):
        """
        Initialize a Neural Network with the provided parameters.

        :param num_epochs: Number of epochs for training
        :param batch_size: Size of the mini-batch
        :param optimizer: Optimizer to use for training
        :param activation: Activation function to use in hidden layers
        :param hidden_sizes: List of sizes for each hidden layer
        :param num_classes: Number of output classes
        :param verbose: If True, print training progress
        :param random_seed: Seed for generating random numbers
        """
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.batch_size = batch_size
        rng = np.random.RandomState(random_seed)
        self.layers = [Layer(size, activation, random_seed=rng.randint(0, 2 ** 32 - 1)) for size in hidden_sizes]
        self.layers.append(Layer(num_classes, 'softmax', random_seed=rng.randint(0, 2 ** 32 - 1)))
        self.num_classes = num_classes

        self.scaler = StandardScaler()

    @staticmethod
    def to_one_hot(y: np.ndarray, num_classes: int):
        """
        Convert label vector to one-hot encoded matrix.

        :param y: A 1D numpy array of labels
        :param num_classes: Number of classes
        :return: A 2D numpy array representing the one-hot encoding
        """
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[range(y.shape[0]), y] = 1
        return one_hot

    def categorical_cross_entropy(self, y_pred: np.ndarray, y_true: np.ndarray, derivative=False) -> Union[
        float, Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate the categorical cross-entropy loss or its derivative.

        :param y_pred: Predicted output
        :param y_true: Actual labels
        :param derivative: If True, calculate the derivative instead
        :return: Loss value (scalar) or derivative of the loss function
        """
        y_true = self.to_one_hot(y_true, self.num_classes)

        # Clip to prevent NaN's and Inf's to prevent log(0) or division by zero:
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if derivative:
            return -y_true / y_pred
            # return 2 * (y_pred - y_true)
        else:
            return -np.sum(y_true * np.log(y_pred), axis=-1)
            # return ((y_true - y_pred) ** 2).sum(-1)

    def _forward(self, X: np.ndarray):
        """
        Performs a forward pass throught the neural network.

        :param X: input data
        :return: the output of the last layer of the neural network, necessary to calculate backprop
        """
        if not self.layers:
            raise ValueError("No layers in the neural network.")

        for layer in self.layers:
            X = layer.forward_pass(X)

        return X

    def _backpropagation(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Executes the backpropagation algorithm.

        The goal of backpropagation is to compute the gradient of the loss function with respect to the weights of the network,
        which is done by propagating the gradient backwards through the network. The gradients are then used to update the weights and biases to minimize the loss
        :param y_pred: network's output from the forward pass algorithm
        :param y_true: true labels
        """
        # Calculate the initial gradient as the derivative of the loss function
        grads = []
        dx = self.categorical_cross_entropy(y_pred, y_true, derivative=True)

        for layer in self.layers[::-1]:
            # Calculate the gradient at the current layer
            dx, dw, db = layer.backward_pass(dx)
            grads.append((dw, db))

        grads = grads[::-1]
        self.optimizer.update(self.layers, grads)  # Update the weights and biases for all layers

    def _run_single_epoch(self, X: np.ndarray, y: np.ndarray, optimize: bool) -> Tuple[float, float]:
        """
        Run one epoch of forward and backward pass, and computes the accuracy and loss for this epoch.

        :param X: The input data
        :param y: The corresponding labels
        :param optimize: If True, performs optimization (backpropagation)
        :return: The average loss and accuracy for this epoch
        """
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for X_batch, y_batch in _create_mini_batches(X, y, self.batch_size, shuffle=optimize, drop_last=optimize):
            y_pred = self._forward(X_batch)

            loss = self.categorical_cross_entropy(y_pred, y_batch)
            total_loss += loss.sum()

            correct = self.count_correct_predictions(y_batch, y_pred)
            correct_predictions += correct

            total_samples += len(X_batch)

            if optimize:
                self._backpropagation(y_pred, y_batch)

        average_loss = total_loss / total_samples
        average_accuracy = correct_predictions / total_samples

        return average_loss, average_accuracy

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None):
        """
        Train the neural network.

        :param X_train: Training input
        :param y_train: Training labels
        :param X_val: Validation input, if available
        :param y_val: Validation labels, if available
        """

        self.history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

        X_train = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val = self.scaler.transform(X_val)

        for epoch in range(self.num_epochs):
            train_loss, train_accuracy = self._run_single_epoch(X_train, y_train, optimize=True)
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_accuracy)

            if X_val is not None and y_val is not None:
                val_loss, val_accuracy = self._run_single_epoch(X_val, y_val, optimize=False)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)

                if self.verbose:
                    print(f'Epoch {epoch + 1}: '
                          f'train_loss={train_loss} train_accuracy={train_accuracy} '
                          f'val_loss={val_loss} val_accuracy={val_accuracy}')

    @staticmethod
    def count_correct_predictions(y_true, y_pred):
        """
        Counts the number of correct predictions.

        :param y_true: true labels
        :param y_pred: predicted labels
        :return: number of correct predictions
        """
        return np.sum(y_true == y_pred.argmax(axis=-1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluates the network's performance on the provided data.

        :param X: The input data
        :return: class predictions
        """
        X = self.scaler.transform(X)
        y_pred = self._forward(X)
        return y_pred.argmax(-1)
