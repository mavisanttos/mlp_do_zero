import numpy as np

class CategoricalCrossEntropy:
    def forward(self, y_pred, y_true):
        """
        Calcula a perda (Loss) usando Cross-Entropy.
        y_pred: matriz (batch_size, num_classes) com as probabilidades do Softmax
        y_true: matriz (batch_size, num_classes) com os rótulos reais em One-Hot
        """
        # Adicionamos um valor minúsculo (1e-15) para evitar log(0), que resultaria em NaN
        y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        
        # Fórmula da Cross-Entropy: - soma(y_true * log(y_pred)) / batch_size
        self.loss = -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]
        
        # Guardamos as previsões e os rótulos reais para o backward
        self.y_pred = y_pred
        self.y_true = y_true
        return self.loss

    def backward(self):
        """
        Derivada simplificada da combinação Softmax + Cross-Entropy.
        Retorna o gradiente em relação aos logits de entrada.
        Dividimos pelo tamanho do batch para que o gradiente seja a média do lote.
        """
        batch_size = self.y_true.shape[0]
        return (self.y_pred - self.y_true) / batch_size