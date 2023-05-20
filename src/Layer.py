from typing import Optional, Any

import numpy as np
from numpy import ndarray


class Layer:
    def __init__(self, hidden_units, activation: Optional[str] = None, random_seed=None):
        """
        Initialize the layer with the specified number of neurons and activation function.

        :param hidden_units: The number of neurons in the layer
        :param activation: The activation function to use (if any). Must be one of [None, 'relu', 'softmax'].
        :param random_seed: Seed for the random number generator, to ensure reproducibility.
        """
        if activation not in [None, 'relu', 'softmax']:
            raise KeyError('wrong activation function')

        self.hidden_units = hidden_units
        self.activation = activation
        self.weights = None
        self.biases = None
        self.random_seed = random_seed if random_seed is not None else np.random.randint(0, 2 ** 32 - 1)

    def _init_params(self, input_size: int, hidden_units: int):
        """
        Initialize the weights and biases for the layer.

        :param input_size: The number of inputs to the layer
        :param hidden_units: The number of neurons in the layer
        """

        self.weights = np.random.RandomState(self.random_seed).randn(input_size, hidden_units) * np.sqrt(
            2. / input_size)
        self.biases = np.zeros((1, hidden_units))

    def _apply_activation(self, weighted_inputs: np.ndarray):
        """
        Apply the activation function to the weighted inputs.

        :param weighted_inputs: The weighted inputs to the layer
        :return: The outputs of the layer after applying the activation function
        """
        if self.activation == 'relu':
            return self.relu(weighted_inputs)
        elif self.activation == 'softmax':
            return self.softmax(weighted_inputs)
        else:
            return weighted_inputs  # just return input

    def _apply_activation_backward(self, weighted_inputs: np.ndarray, grad: np.ndarray):
        """
        Apply the derivative of the activation function to the weighted inputs, for use in backpropagation.

        :param weighted_inputs: The weighted inputs to the layer
        :param grad: The gradient of the loss function with respect to the outputs of the layer
        :return: The gradient of the loss function with respect to the weighted inputs
        """
        if self.activation == 'relu':
            return self.relu_backward(weighted_inputs, grad)
        elif self.activation == 'softmax':
            return self.softmax_backward(weighted_inputs, grad)
        else:
            return grad  # just return grad

    def forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass of the data through the layer.

        :param inputs: The input data
        :return: The outputs of the layer
        """
        self.inputs = inputs
        if self.weights is None:
            self._init_params(inputs.shape[-1], self.hidden_units)

        self.weighted_inputs = inputs @ self.weights + self.biases
        return self._apply_activation(self.weighted_inputs)

    def backward_pass(self, next_layer_grad: np.ndarray) -> tuple[Any, Any, ndarray]:
        """
        Perform a backward pass (backpropagation) of the gradient through the layer.

        :param next_layer_grad: The gradient of the loss function with respect to the outputs of the next layer
        :return: The gradient of the loss function with respect to the inputs of the layer
        """
        da = self._apply_activation_backward(self.weighted_inputs, next_layer_grad)
        db = np.sum(da, axis=0)
        dw = self.inputs.T @ da
        dx = da @ self.weights.T
        return dx, dw, db

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        s = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return s

    @staticmethod
    def softmax_backward(x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        s = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

        diag = np.stack([
            np.diag(s[i])
            for i in range(len(s))
        ], 0)
        softmax_grad = diag - np.einsum('bi,bj->bij', s, s)
        # grad: batch_size x class_num
        # softmax_grad: batch_size x class_num x class_num
        return np.einsum('bc,bcd->bd', grad, softmax_grad)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_backward(x, grad):
        relu_grad = (x >= 0).astype(x.dtype)
        return relu_grad * grad
