import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        """
        Inicializa o otimizador Stochastic Gradient Descent.
        learning_rate: dita o tamanho do passo na direção oposta ao gradiente.
        """
        self.learning_rate = learning_rate

    def update(self, layers):
        """
        Varre todas as camadas da rede e atualiza pesos e biases se eles existirem.
        As camadas que possuem pesos devem expor atributos .W, .b, .dW e .db.
        """
        for layer in layers:
            # Nem todas as camadas têm parâmetros treináveis (ex: ativações)
            if hasattr(layer, 'W'):
                # Atualização clássica do SGD: W = W - lr * dW
                layer.W -= self.learning_rate * layer.dW
                layer.b -= self.learning_rate * layer.db