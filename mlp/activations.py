import numpy as np

class ReLU:
    def forward(self, x):
        """
        Aplica a função ReLU: f(x) = max(0, x)
        Salva o input para usar no cálculo da derivada depois.
        """
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, output_gradient):
        """
        Derivada da ReLU: 1 se x > 0, senão 0.
        Multiplica-se pelo gradiente que vem da camada seguinte (chain rule).
        """
        return output_gradient * (self.input > 0)

class Softmax:
    def forward(self, x):
        """
        Aplica Softmax de forma numericamente estável.
        Transforma scores (logits) em probabilidades que somam 1.
        """
        # Subtrair o max evita overflow exponencial (inf)
        exp_shifted = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.output = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
        return self.output
    
    def backward(self, output_gradient):
        """
        Nota: A derivada do Softmax combinada com a Cross-Entropy fica simplificada.
        Se usada isoladamente, a derivada do Softmax é uma matriz Jacobiana.
        Para o nosso fluxo padrão, vamos gerenciar o cálculo simplificado na Loss.
        """
        # Retornamos o gradiente recebido puro, pois a simplificação será feita na classe Loss
        return output_gradient