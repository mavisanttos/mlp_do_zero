# Multi-Layer Perceptron (MLP) do Zero

&emsp; Este repositório contém a implementação completa de uma rede neural artificial do tipo Multi-Layer Perceptron (MLP) construída puramente em Python utilizando apenas a biblioteca **NumPy** para operações matriciais. O modelo foi treinado e validado no conjunto de dados MNIST para classificação de dígitos manuscritos.

---

## Como Rodar

### 1. Clonar o repositório e acessar a pasta

```bash
git clone https://github.com/mavisanttos/mlp_do_zero.git
cd mlp_do_zero
```

### 2. Instalar as dependências

Certifique-se de ter o Python 3 instalado. Instale os pacotes necessários via pip:

```bash
pip install -r requirements.txt
```

### 3. Executar os Experimentos

Abra o Jupyter Notebook para visualizar o treinamento e os gráficos comparativos:

```bash
jupyter notebook notebooks/experimentos.ipynb
```

---

## Arquitetura

&emsp; Para atingir e superar a meta de acurácia de 92%, estruturei o **Modelo A** (configuração principal) com a seguinte arquitetura sequencial:

- **Camada de Entrada:** 784 neurônios (correspondentes às imagens de $28 \times 28$ pixels achatadas).
- **Camada Oculta 1:** 128 neurônios + Ativação ReLU.
- **Camada Oculta 2:** 64 neurônios + Ativação ReLU.
- **Camada de Saída:** 10 neurônios (um para cada dígito de 0 a 9) + Ativação Softmax.

**Justificativa:**

&emsp; A escolha da ReLU para as camadas ocultas mitiga o problema do sumiço do gradiente (*vanishing gradient*), comum em funções como a Sigmoide, acelerando a convergência. O Softmax na saída é a escolha padrão para classificação multiclasse, pois transforma os scores brutos (logits) em uma distribuição de probabilidade estável onde a soma de todas as saídas é igual a 1.

---

## Resultados e Experimentos

&emsp; Para garantir a reprodutibilidade dos testes, foi fixada uma semente de aleatoriedade (`np.random.seed(42)`) no início dos experimentos. Foram comparadas duas configurações para avaliar o comportamento do aprendizado:

| Métrica / Parâmetro | Modelo A (Principal) | Modelo B (Comparativo) |
| :--- | :--- | :--- |
| **Arquitetura** | Entrada -> 128 -> 64 -> Saída | Entrada -> 32 -> 16 -> Saída |
| **Learning Rate ($\eta$)** | 0.1 | 0.01 |
| **Tamanho do Batch** | 64 | 64 |
| **Épocas** | 15 | 15 |
| **Acurácia Final (Teste)**| **97.65%** | **95.17%** |

### Curvas de Treinamento

&emsp; O gráfico com o comportamento da função de perda (Loss) e a evolução da acurácia ao longo das épocas foi gerado automaticamente e está salvo na pasta `results/curvas_treinamento.png`. 

* **Modelo A:** Demonstrou uma convergência extremamente agressiva e excelente capacidade de generalização, superando com folga a meta mínima e estacionando em **97.65%** de acurácia.
* **Modelo B:** Embora utilize uma rede bem mais enxuta e uma taxa de aprendizado 10 vezes menor ($\eta = 0.01$), **o modelo conseguiu atingir a meta estipulada de 92%** logo na 5ª época (registrando 92.56%) e finalizou o treino com **95.17%**. Isso demonstra que, mesmo com menos capacidade representativa, o gradiente continuou fluindo e refinando os parâmetros, embora em um ritmo visivelmente mais lento que o Modelo A.

### Análise de Erros (Matriz de Confusão)

&emsp; A matriz de confusão gerada no teste do **Modelo A** (`results/matriz_confusao.png`) revela que a diagonal principal concentra a esmagadora maioria das predições, comprovando a eficácia do modelo. No entanto, uma análise detalhada dos erros revela padrões de confusão muito semelhantes aos desvios de interpretação humanos:
* **4 vs. 9:** É a maior fonte de erro da rede. O traço superior aberto do 4 muitas vezes se assemelha ao laço fechado do 9 no padrão manuscrito rápido.
* **3 vs. 5:** Comum devido à metade inferior de ambos os números possuir exatamente a mesma curvatura anatômica.
* **7 vs. 2:** Ocorre em casos onde o traço do topo do 7 é feito com uma inclinação acentuada ou base curvada, confundindo as ativações lineares.

### Validação Matemática (Gradient Check)

&emsp; Para garantir que a implementação analítica do Backpropagation no NumPy não continha erros sutis de sinal ou de derivada, foi criado um script de **Gradient Checking** numérico utilizando a aproximação de diferença central:

$$\frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon}$$

&emsp; Utilizando um $\epsilon = 10^{-5}$ em uma topologia reduzida, o erro relativo calculado entre o gradiente analítico e o numérico foi de **aproximadamente $1.4 \times 10^{-9}$**. Sendo consideravelmente menor que o limite padrão tolerado ($10^{-5}$), o teste valida matematicamente que o fluxo de gradientes da rede está calculando e atualizando os pesos de forma 100% correta.

---

## Decisões e Dificuldades

### 1. Qual foi a decisão técnica mais difícil que você tomou? Por que fez essa escolha?

A decisão mais complexa foi a modelagem algébrica das matrizes durante o passo de retropropagação (Backpropagation). Garantir que os produtos escalares (`np.dot`) entre matrizes de dimensões distintas — como alinhar os gradientes vindos da camada posterior com as ativações salvas no Forward — estivessem corretos exigiu atenção à álgebra linear.

Optei por fazer o casamento analítico da derivada da Cross-Entropy com o Softmax, o que resultou na simplificação $\hat{y} - y$. Essa escolha evitou o cálculo exaustivo de matrizes Jacobianas isoladas, reduzindo drasticamente a complexidade do código e o tempo de execução por batch.

### 2. O que você tentou que não funcionou? O que aprendeu com isso?

No início do desenvolvimento, tentei inicializar os pesos com matrizes preenchidas com zeros (`np.zeros`) ou valores aleatórios sem escala. Como resultado prático, a rede simplesmente não aprendia — a Loss ficava estagnada. Com isso, compreendi o problema da **quebra de simetria**: se todos os neurônios começam iguais, todos atualizam do mesmo jeito e a rede age como se tivesse apenas um neurônio por camada.

Para corrigir isso, pesquisei e implementei a **Inicialização He** (`np.sqrt(2.0 / input_dim)`), ideal para a ReLU, o que desbloqueou o aprendizado e estabilizou os gradientes.

Outro problema foi a ocorrência de erros do tipo `NaN` no cálculo da Loss. Descobri que o `np.log` quebrava quando a rede previa exatamente 0% de chance para uma classe correta. Resolvi isso aplicando um `np.clip(y_pred, 1e-15, 1.0 - 1e-15)` para impedir valores estritamente zero.

### 3. Se fosse refazer do zero, o que faria diferente?

Se eu fosse reestruturar o projeto, implementaria uma classe abstrata `Layer` para servir de base para todas as camadas (`Dense`, `ReLU`, `Softmax`), padronizando estritamente as assinaturas dos métodos `forward` e `backward`. Embora a verificação dinâmica usando `hasattr(layer, 'W')` no otimizador tenha funcionado para este projeto, uma estrutura de herança deixaria o código ainda mais próximo do design pattern utilizado em bibliotecas profissionais como o PyTorch.