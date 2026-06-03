# RoadMap do Projeto

## Marco 1: Protótipo de Otimização Hexagonal & Predição Espaço-Temporal

### Fase 1: Engenharia de Dados & Alinhamento Territorial
- **Objetivo:** Filtrar, limpar e imputar a base de Fortaleza de forma que tenhamos latitude, longitude, data, tipo_evento, bairro e ais 100% preenchidos e válidos.
- **Tarefas:**
  1. Carregar a base de dados enriquecida.
  2. Implementar script para mapeamento de Bairro para AIS.
  3. Resolver nulos do campo `ais` usando o mapeamento e vizinho mais próximo geográfico.
  4. Salvar base limpa e enriquecida em `data/processed/fortaleza_crimes.csv`.

### Fase 2: Definição da Área de Interesse (AoI) e Grade Hexagonal Dinâmica
- **Objetivo:** Definir a área de interesse baseada nos hotspots de crimes e gerar uma grade hexagonal parametrizável por rotação, deslocamento e raio.
- **Tarefas:**
  1. Definir envelope convexo ou delimitação geográfica dos crimes de Fortaleza (AoI).
  2. Implementar classe de geração de grade hexagonal com parâmetros ($dx, dy, \theta, R$).
  3. Implementar função para indexar cada ocorrência de crime dentro de sua célula hexagonal correspondente.

### Fase 3: Pipeline de Previsão Semanal por Regressão Linear Múltipla
- **Objetivo:** Estruturar a série temporal semanal por unidade territorial e rodar a Regressão Linear Múltipla (RLM).
- **Tarefas:**
  1. Criar agregador semanal de ocorrências por unidade geográfica.
  2. Desenvolver preditor RLM com variáveis de lag temporal (ex: lags t-1, t-2, t-3).
  3. Implementar validação temporal (ex: train/test split cronológico ou cross-validation walk-forward) e cálculo do Erro Quadrático Médio (EQM).

### Fase 4: Algoritmo Genético de Busca do Grid Ótimo
- **Objetivo:** Otimizar as variáveis do grid ($dx, dy, \theta, R$) usando Algoritmo Genético.
- **Tarefas:**
  1. Definir representação cromossômica e limites dos parâmetros.
  2. Implementar a função de fitness (treinar RLM na grade gerada e retornar o EQM global).
  3. Implementar operadores genéticos (seleção, crossover aritmético/heurístico, mutação gaussiana/uniforme).
  4. Executar o AG e extrair a melhor configuração do mosaico hexagonal.

### Fase 5: Avaliação Comparativa Final e Relatório
- **Objetivo:** Consolidar a comparação de erro entre os modelos.
- **Tarefas:**
  1. Calcular o EQM de previsão semanal agrupando por Bairros.
  2. Calcular o EQM de previsão semanal agrupando por AIS.
  3. Calcular o EQM utilizando o Mosaico Hexagonal Ótimo encontrado pelo AG.
  4. Gerar relatórios e gráficos comparativos de desempenho.
