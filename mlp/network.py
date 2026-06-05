import numpy as np

class Dense:
    def __init__(self, input_dim, output_dim):
        """
        Inicialização He (Xavier adaptada para ReLU):
        Divide por raiz(2 / input_dim) para manter a variância do gradiente estável.
        """
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((1, output_dim))
        
        # Espaços para guardar os gradientes calculados no backward
        self.dW = None
        self.db = None
        self.input = None

    def forward(self, x):
        """
        Z = X * W + b
        """
        self.input = x  # Guardamos o input (X), pois ele é usado para calcular dW no backward
        return np.dot(x, self.W) + self.b

    def backward(self, output_gradient):
        """
        Calcula os gradientes da camada e retorna o gradiente para a camada anterior.
        """
        # dW = X^T * dZ (Média do batch)
        self.dW = np.dot(self.input.T, output_gradient)
        # db = soma de dZ ao longo do batch
        self.db = np.sum(output_gradient, axis=0, keepdims=True)
        
        # Gradiente que vai para a camada anterior: dZ * W^T
        input_gradient = np.dot(output_gradient, self.W.T)
        return input_gradient


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        """Adiciona uma camada (Dense, ReLU, Softmax, etc) na sequência."""
        self.layers.append(layer)

    def forward(self, x):
        """Passa o dado sequencialmente por todas as camadas."""
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, loss_gradient):
        """Passa o gradiente de trás para frente (Backpropagation)."""
        grad = loss_gradient
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad