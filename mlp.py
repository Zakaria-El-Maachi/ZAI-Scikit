import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def forward(self, input_data):
        pass

    @abstractmethod
    def backward(self, output_gradient, learning_rate):
        pass


class FullyConnectedLayer(Layer):
    def __init__(self, input_size, output_size, use_bias=True):
        self.use_bias = use_bias
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        self.bias = np.random.randn(output_size, 1) if use_bias else None

    def forward(self, input_data):
        self.input_data = input_data
        self.output_data = np.dot(self.weights, input_data)
        if self.use_bias:
            self.output_data += self.bias
        return self.output_data

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input_data.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        
        self.weights -= learning_rate * weights_gradient
        if self.use_bias:
            self.bias -= learning_rate * output_gradient
        return input_gradient




class ActivationFunction(Layer):
    def __init__(self, function, function_prime):
        self.function = function
        self.function_prime = function_prime

    def forward(self, input_data):
        self.input_data = input_data
        return self.function(input_data)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.function_prime(self.input_data)

class Sigmoid(ActivationFunction):
    def __init__(self):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        sigmoid_prime = lambda x: sigmoid(x) * (1 - sigmoid(x))
        super().__init__(sigmoid, sigmoid_prime)

class ReLU(ActivationFunction):
    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        relu_prime = lambda x: (x > 0).astype(float)
        super().__init__(relu, relu_prime)

class Softmax(ActivationFunction):
    def __init__(self):
        softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
        softmax_prime = lambda x: softmax(x) * (1 - softmax(x))
        super().__init__(softmax, softmax_prime)





class MLP:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, output_gradient, learning_rate):
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, learning_rate)

    def fit(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            for x, target in zip(X, y):
                output = self.forward(x)
                output_gradient = self.loss_derivative(output, target)
                self.backward(output_gradient, learning_rate)

    def predict(self, X):
        return [self.forward(x) for x in X]

    def loss(self, output, target):
        return np.mean(np.square(output - target))

    def loss_derivative(self, output, target):
        return 2 * (output - target) / output.size
