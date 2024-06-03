import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    """
    Abstract base class for a layer in a neural network.
    """
    @abstractmethod
    def forward(self, input_data):
        """
        Perform the forward pass computation.

        Parameters:
        input_data (numpy.ndarray): Input data to the layer.

        Returns:
        numpy.ndarray: Output data from the layer.
        """
        pass

    @abstractmethod
    def backward(self, output_gradient, learning_rate):
        """
        Perform the backward pass computation.

        Parameters:
        output_gradient (numpy.ndarray): Gradient of the loss with respect to the layer's output.
        learning_rate (float): Learning rate for the layer's parameter update.

        Returns:
        numpy.ndarray: Gradient of the loss with respect to the layer's input.
        """
        pass

class FullyConnectedLayer(Layer):
    """
    Fully connected (dense) layer in a neural network.
    
    Parameters:
    input_size (int): Number of input features.
    output_size (int): Number of output features.
    use_bias (bool): Whether to include a bias term. Default is True.
    """
    def __init__(self, input_size, output_size, use_bias=True):
        self.use_bias = use_bias
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((output_size, 1)) if use_bias else None

    def forward(self, input_data):
        """
        Perform the forward pass computation.

        Parameters:
        input_data (numpy.ndarray): Input data to the layer.

        Returns:
        numpy.ndarray: Output data from the layer.
        """
        self.input_data = input_data
        self.output_data = np.dot(self.weights, input_data)
        if self.use_bias:
            self.output_data += self.bias
        return self.output_data

    def backward(self, output_gradient, learning_rate):
        """
        Perform the backward pass computation.

        Parameters:
        output_gradient (numpy.ndarray): Gradient of the loss with respect to the layer's output.
        learning_rate (float): Learning rate for the layer's parameter update.

        Returns:
        numpy.ndarray: Gradient of the loss with respect to the layer's input.
        """
        weights_gradient = np.dot(output_gradient, self.input_data.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        
        # Gradient clipping
        np.clip(weights_gradient, -1, 1, out=weights_gradient)
        np.clip(input_gradient, -1, 1, out=input_gradient)
        
        self.weights -= learning_rate * weights_gradient
        if self.use_bias:
            self.bias -= learning_rate * np.sum(output_gradient, axis=1, keepdims=True)
        return input_gradient

class ActivationFunction(Layer):
    """
    Activation function layer in a neural network.
    
    Parameters:
    function (callable): The activation function.
    function_prime (callable): The derivative of the activation function.
    """
    def __init__(self, function, function_prime):
        self.function = function
        self.function_prime = function_prime

    def forward(self, input_data):
        """
        Perform the forward pass computation.

        Parameters:
        input_data (numpy.ndarray): Input data to the layer.

        Returns:
        numpy.ndarray: Output data from the layer.
        """
        self.input_data = input_data
        return self.function(input_data)

    def backward(self, output_gradient, learning_rate):
        """
        Perform the backward pass computation.

        Parameters:
        output_gradient (numpy.ndarray): Gradient of the loss with respect to the layer's output.
        learning_rate (float): Learning rate for the layer's parameter update.

        Returns:
        numpy.ndarray: Gradient of the loss with respect to the layer's input.
        """
        input_prime = self.function_prime(self.input_data)
        output = output_gradient * input_prime
        np.clip(output, -1, 1, out=output)  # Gradient clipping
        return output

class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function layer.
    """
    def __init__(self):
        sigmoid = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clipping to avoid overflow
        sigmoid_prime = lambda x: sigmoid(x) * (1 - sigmoid(x))
        super().__init__(sigmoid, sigmoid_prime)

class ReLU(ActivationFunction):
    """
    ReLU activation function layer.
    """
    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        relu_prime = lambda x: (x > 0).astype(float)
        super().__init__(relu, relu_prime)

class Softmax(ActivationFunction):
    """
    Softmax activation function layer.
    """
    def __init__(self):
        softmax = lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=0)
        super().__init__(softmax, lambda x: x)  # Softmax prime is handled differently in practice

class MLP:
    """
    Multi-Layer Perceptron (MLP) model.
    
    Parameters:
    epochs (int): Number of training epochs. Default is 500.
    learning_rate (float): Learning rate for updating the MLP's parameters. Default is 0.01.
    """
    def __init__(self, epochs=500, learning_rate=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.layers = []

    def add_layer(self, layer):
        """
        Add a layer to the MLP.

        Parameters:
        layer (Layer): The layer to add.
        """
        self.layers.append(layer)

    def forward(self, input_data):
        """
        Perform the forward pass computation.

        Parameters:
        input_data (numpy.ndarray): Input data to the MLP.

        Returns:
        numpy.ndarray: Output data from the MLP.
        """
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, output_gradient, learning_rate):
        """
        Perform the backward pass computation.

        Parameters:
        output_gradient (numpy.ndarray): Gradient of the loss with respect to the MLP's output.
        learning_rate (float): Learning rate for updating the MLP's parameters.
        """
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, learning_rate)

    def fit(self, X, y):
        """
        Train the MLP using the provided training data.

        Parameters:
        X (numpy.ndarray): Training features.
        y (numpy.ndarray): Training labels.
        """
        for epoch in range(self.epochs):
            for x, target in zip(X, y):
                x = x.reshape(-1, 1)  # Ensure column vector
                target = target.reshape(-1, 1)  # Ensure column vector
                output = self.forward(x)
                output_gradient = self.loss_derivative(output, target)
                self.backward(output_gradient, self.learning_rate)

    def predict(self, X):
        """
        Predict the outputs for the given input data.

        Parameters:
        X (numpy.ndarray): Input data.

        Returns:
        list: Predicted outputs for the input data.
        """
        outputs = [self.forward(x.reshape(-1, 1)) for x in X]
        return outputs

    def loss(self, output, target):
        """
        Calculate the loss between the predicted and target values.

        Parameters:
        output (numpy.ndarray): Predicted values.
        target (numpy.ndarray): Target values.

        Returns:
        float: Loss value.
        """
        return np.mean(np.square(output - target))

    def loss_derivative(self, output, target):
        """
        Calculate the derivative of the loss function.

        Parameters:
        output (numpy.ndarray): Predicted values.
        target (numpy.ndarray): Target values.

        Returns:
        numpy.ndarray: Derivative of the loss with respect to the output.
        """
        return 2 * (output - target) / output.size