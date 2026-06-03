# Plano de Fase: Fase 4 - Algoritmo Genético de Busca do Grid Ótimo

## Objetivo
Implementar o Algoritmo Genético (AG) para encontrar os parâmetros ideais do grid hexagonal móvel. Os indivíduos serão representados por um vetor de números reais (cromossomo de 4 genes: $dx, dy, \theta, R$). A aptidão (fitness) será a minimização do Erro Quadrático Médio (EQM) global da Regressão Linear Múltipla (RLM) calculada sobre a grade resultante.

---

## Modificações Propostas

### 1. Script do Algoritmo Genético
#### [NEW] [src/genetic_algorithm.py](file:///C:/Users/Boanerges/Desktop/Projetos/Crime_Predict-Algoritmo%20Genetico/src/genetic_algorithm.py)
Este script conterá a estrutura evolucionária principal:
- **População Inicial**: Indivíduos com valores aleatórios dentro de intervalos lógicos:
  - $dx, dy \in [0.0, 1.0]$ (translação proporcional ao raio)
  - $\theta \in [0, \pi/3]$ (rotação local do hexágono)
  - $R \in [0.005, 0.02]$ (tamanho do hexágono em graus, controlando a granularidade espacial)
- **Função Fitness**: Executar `HexagonalGrid` com os parâmetros, mapear os crimes para os hexágonos gerados, criar a série temporal com `prepare_weekly_series`, avaliar com `evaluate_model` e retornar $1 / (EQM + 1e-6)$ para maximizar a aptidão (ou simplesmente minimizar o EQM diretamente).
- **Operadores Genéticos**:
  - Seleção por torneio.
  - Crossover aritmético simples (média ponderada dos pais).
  - Mutação Gaussiana com decaimento de variância ao longo das gerações.
  - Elitismo (preservação do melhor indivíduo de cada geração).

---

## Plano de Verificação

### Teste Automatizado
- Executar um teste rápido do AG rodando por poucas gerações (ex: 3 gerações, população de 5 indivíduos) para garantir que a convergência de código funciona sem gargalos de memória e que o melhor indivíduo de fato reduz o EQM inicial.

---

## Passos para Execução
1. Criar `src/genetic_algorithm.py`.
2. Executar e validar com script piloto.
3. Atualizar o `STATE.md`.
