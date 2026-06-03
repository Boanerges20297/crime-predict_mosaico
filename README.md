# Busca evolucionária pelo mosaico hexagonal regular ótimo para previsão espaço-temporal de crimes em Fortaleza/CE

Este projeto implementa um **Algoritmo Genético (AG)** para encontrar o particionamento ótimo do território da Cidade de Fortaleza/CE em uma grade de hexágonos regulares parametrizada. O objetivo de otimização é encontrar a divisão espacial que minimiza o erro de previsão do número de crimes semanais em cada célula territorial por meio de um modelo preditivo de **Regressão Linear Múltipla (RLM)**.

---

## 🗺️ Mapa Interativo

Geramos um mapa completo no arquivo [mapa_interativo_fortaleza.html](file:///C:/Users/Boanerges/Desktop/Projetos/Crime_Predict-Algoritmo%20Genetico/mapa_interativo_fortaleza.html) que permite alternar as camadas visuais para comparar espacialmente:
1. **Áreas Integradas de Segurança (AIS)**
2. **Bairros**
3. **Mosaico Hexagonal Ótimo** (com intensidade de cores baseada na contagem de crimes)

---

## 🧬 Como funciona o Algoritmo Genético

O objetivo é definir as coordenadas e dimensões de uma grade hexagonal de forma a melhorar a performance preditiva de séries temporais semanais de crime.

### 1. Representação do Cromossomo (Genes)
Cada indivíduo da população é representado por um cromossomo de **4 genes de valores reais**:
- $dx$: Deslocamento horizontal inicial (fração entre $0.0$ e $1.0$ do raio $R$).
- $dy$: Deslocamento vertical inicial (fração entre $0.0$ e $1.0$ do raio $R$).
- $\theta$: Ângulo de rotação da grade hexagonal, variando de $0$ a $\pi/3$ radianos ($0^\circ$ a $60^\circ$).
- $R$: Raio do hexágono regular (distância do centro a um dos vértices), limitado de $0.005$ a $0.02$ graus geográficos (~$550$m a ~$2200$m).

### 2. Função de Aptidão (Fitness)
A aptidão mede o quão adequada é a grade gerada por aquele indivíduo. A função fitness executa os seguintes passos:
1. Gera a grade hexagonal com os parâmetros do cromossomo.
2. Associa cada crime geolocalizado ($latitude, longitude$) à sua respectiva célula hexagonal através de uma busca indexada rápida em árvore espacial R-Tree (`shapely.strtree.STRtree`).
3. Agrupa a contagem de crimes semanalmente para cada célula ativa.
4. Gera variáveis de lag temporal ($t-1$, $t-2$, $t-3$) para cada série.
5. Executa um treinamento Walk-Forward cronológico: dados até 31/12/2025 para treino e dados de 2026 para teste do modelo de RLM.
6. Calcula o Erro Quadrático Médio (EQM) de teste e retorna:
$$\text{Aptidão} = \frac{1}{\text{EQM} + 1e-8}$$

### 3. Operadores Genéticos
- **Seleção**: Seleção por torneio de tamanho 3 (sorteia 3 indivíduos aleatoriamente e o mais apto é escolhido para reprodução).
- **Crossover (Cruzamento)**: Cruzamento aritmético linear simples. A partir de dois pais ($P_1, P_2$) com probabilidade de $80\%$, gera dois filhos ($F_1, F_2$) através de:
$$F_1 = \alpha P_1 + (1-\alpha) P_2$$
$$F_2 = (1-\alpha) P_1 + \alpha P_2$$
onde $\alpha$ é um valor uniforme aleatório em $[0.1, 0.9]$.
- **Mutação**: Mutação Gaussiana aplicada a genes individuais com probabilidade de $15\%$. Para acelerar o ajuste refinado nas últimas gerações, a força do ruído gaussiano decai gradativamente:
$$\sigma_{t} = 0.1 \times \left(1.0 - \frac{t}{T}\right)$$
onde $t$ é a geração atual e $T$ é o total de gerações.
- **Elitismo**: O melhor indivíduo absoluto de cada geração é preservado e repassado intocado à geração subsequente.

---

## 📈 Resultados e Comparativo

Ao rodar o modelo preditivo nas divisões clássicas versus a grade ótima encontrada:

| Divisão Territorial | Regiões Ativas | Erro Quadrático Médio (EQM) |
| :--- | :---: | :---: |
| **Áreas Integradas de Segurança (AIS)** | 14 | **437.1137** |
| **Bairros** | 162 | **16.2539** |
| **Mosaico Hexagonal Ótimo** | 323 | **6.7456** |

---

## 🚀 Como Executar o Projeto

1. Instale as dependências:
   ```bash
   pip install pandas numpy shapely scikit-learn folium
   ```
2. Processe a base inicial (Fase 1):
   ```bash
   python src/prepare_data.py
   ```
3. Execute o algoritmo genético para otimizar os hexágonos (Fase 4):
   ```bash
   python src/run_ga.py
   ```
4. Gere a comparação estatística e relatório comparativo (Fase 5):
   ```bash
   python src/evaluate_comparison.py
   ```
5. Compile o mapa geográfico interativo:
   ```bash
   python src/generate_map.py
   ```
