# Plano de Fase: Fase 2 - Definição da Área de Interesse (AoI) e Grade Hexagonal Dinâmica

## Objetivo
Definir a Área de Interesse (AoI) de Fortaleza/CE (definindo os limites geográficos/hotspots dos crimes) e construir um gerador de grade hexagonal móvel/regular cujos parâmetros (deslocamento $dx, dy$, rotação $\theta$ e tamanho do raio $R$) possam ser ajustados dinamicamente para o Algoritmo Genético. Além disso, estruturar a lógica que mapeia os crimes da base tratada para o ID de sua célula hexagonal correspondente.

---

## Modificações Propostas

### 1. Script utilitário de geração de grade hexagonal
#### [NEW] [src/grid_generator.py](file:///C:/Users/Boanerges/Desktop/Projetos/Crime_Predict-Algoritmo%20Genetico/src/grid_generator.py)
Esta classe terá as seguintes funções:
- Definir a caixa delimitadora (bounding box) da AoI usando os quantis (ex: 1% a 99%) ou limites extremos de latitude/longitude das ocorrências.
- Gerar um conjunto de pontos centrais e polígonos correspondentes a hexágonos regulares com raio $R$ (em graus ou metros aproximados), rotacionados em $\theta$ radianos, e deslocados por ($dx, dy$).
- Atribuir a cada ocorrência o identificador da célula hexagonal em que ela cai, calculando de forma eficiente a indexação espacial (sem requerer loop pesado ponto-a-polígono, mas sim usando equações analíticas de grade hexagonal se possível, ou indexação espacial otimizada com Shapely/STRtree).

---

## Plano de Verificação

### Teste Automatizado
- Executar um script que gera uma grade com parâmetros arbitrários (ex: $dx=0, dy=0, \theta=0, R=0.01$).
- Validar se 100% dos pontos de Fortaleza caem dentro ou fora de alguma célula hexagonal e extrair a lista de células ativas (hexágonos que contêm pelo menos 1 crime).
- Verificar se a rotação e deslocamento alteram de fato os identificadores das ocorrências.

---

## Passos para Execução
1. Criar `src/grid_generator.py`.
2. Criar script de teste e verificação.
3. Atualizar o `STATE.md`.
