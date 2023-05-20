from typing import List, Tuple

import numpy as np

from src import Layer


class Optimizer:
    """
    This is the base Optimizer class. It has a learning rate parameter which is set at initialization.
    The update method is intended to be overridden by subclasses that implement specific optimization algorithms.
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        """
        This method is intended to be overridden by subclasses to implement parameter update logic.
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    The SGD class implements Stochastic Gradient Descent with momentum. It inherits from the base Optimizer class.
    It has a momentum parameter which is set at initialization. The update method updates each layer's parameters based
    on the gradients and the learning rate, taking momentum into account.
    """

    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity: List[Tuple[np.ndarray, np.ndarray]] = []

    def update(self, layers, grads):
        """
        Updates the parameters of each layer based on the gradients, the learning rate, and momentum.
        """
        if not self.velocity:
            self.velocity = [(np.zeros_like(layer.weights), np.zeros_like(layer.biases)) for layer in layers]

        for (v_w, v_b), layer, (dw, db) in zip(self.velocity, layers, grads):
            v_w *= self.momentum
            v_w += self.learning_rate * dw
            layer.weights -= v_w

            v_b *= self.momentum
            v_b += self.learning_rate * db
            layer.biases -= v_b


class AdaGrad(Optimizer):
    """
    The AdaGrad class implements the AdaGrad optimization algorithm. It inherits from the base Optimizer class.
    The update method updates each layer's parameters based on the gradients and an adaptive learning rate.
    """

    def __init__(self, learning_rate=0.01, epsilon=1e-7):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.accumulated_grads: List[Tuple[np.ndarray, np.ndarray]] = []

    def update(self, layers: List[Layer], grads: List[Tuple[np.ndarray, np.ndarray]]):
        """
        Updates the parameters of each layer based on the gradients, the learning rate, and an adaptive term.
        :param layers: list of layers
        :param grads: list of gradients for each layer's parameters
        """
        if not self.accumulated_grads:
            self.accumulated_grads = [(np.zeros_like(layer.weights), np.zeros_like(layer.biases)) for layer in layers]

        for (h_w, h_b), layer, (dw, db) in zip(self.accumulated_grads, layers, grads):
            h_w += dw * dw
            layer.weights -= self.learning_rate * dw / (np.sqrt(h_w) + self.epsilon)

            h_b += db * db
            layer.biases -= self.learning_rate * db / (np.sqrt(h_b) + self.epsilon)


class Adam(Optimizer):
    """
    The Adam class implements the Adam optimization algorithm.
    It inherits from the base Optimizer class. Based on
    https://optimization.cbe.cornell.edu/index.php?title=Adam
    """

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7):
        """
        Initialize Adam optimizer.

        :param learning_rate: learning rate
        :param beta1: The exponential decay rate for the first moment estimates
        :param beta2: The exponential decay rate for the second-moment estimates
        :param epsilon: small value to prevent division by zero
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: List[Tuple[np.ndarray, np.ndarray]] = []
        self.v: List[Tuple[np.ndarray, np.ndarray]] = []
        self.t = 0

    def update(self, layers: List[Layer], grads: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        Perform the Adam update on parameters.

        :param layers: list of layers with parameters to update
        :param grads: list of gradients for each layer's parameters
        """
        if not self.m:
            self.m = [[np.zeros_like(layer.weights), np.zeros_like(layer.biases)] for layer in layers]
            self.v = [[np.zeros_like(layer.weights), np.zeros_like(layer.biases)] for layer in layers]

        self.t += 1

        for (m, v), layer, (dw, db) in zip(zip(self.m, self.v), layers, grads):
            m[0] *= self.beta1
            m[0] += (1.0 - self.beta1) * dw
            bias_corrected_first_moment = m[0] / (1.0 - self.beta1 ** self.t)
            v[0] *= self.beta2
            v[0] += (1.0 - self.beta2) * dw ** 2
            bias_corrected_second_moment = v[0] / (1.0 - self.beta2 ** self.t)

            layer.weights -= self.learning_rate * bias_corrected_first_moment / (
                    np.sqrt(bias_corrected_second_moment) + self.epsilon)

            m[1] *= self.beta1
            m[1] += (1.0 - self.beta1) * db
            bias_corrected_first_moment = m[1] / (1.0 - self.beta1 ** self.t)
            v[1] *= self.beta2
            v[1] += (1.0 - self.beta2) * db ** 2
            bias_corrected_second_moment = v[1] / (1.0 - self.beta2 ** self.t)

            layer.biases -= self.learning_rate * bias_corrected_first_moment / (
                    np.sqrt(bias_corrected_second_moment) + self.epsilon)
