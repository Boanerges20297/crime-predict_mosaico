# Projeto: Busca Evolucionária por Mosaico Hexagonal para Previsão de Crimes

## Objetivo
Encontrar a divisão espacial em grade hexagonal regular ótima para a previsão espaço-temporal de crimes em Fortaleza/CE, utilizando um Algoritmo Genético (AG) para otimização dos parâmetros da grade e Regressão Linear Múltipla (RLM) como modelo de previsão. O desempenho do mosaico ótimo será comparado com as divisões político-administrativas oficiais de Bairros e Áreas Integradas de Segurança (AIS).

## Stack Tecnológico
- **Linguagem:** Python 3.14+
- **Processamento de Dados:** Pandas, Numpy
- **Geoprocessamento:** Shapely, H3 (ou grade geométrica customizada), Geopandas (se necessário para visualização)
- **Modelagem Preditiva:** Scikit-Learn (LinearRegression)
- **Otimização:** Algoritmo Genético customizado (Crossover, Mutação, Seleção por Torneio/Roleta)
- **Visualização:** Matplotlib, Seaborn

## Fontes de Dados
1. **Dados Originais:** `data/raw/dados_status.csv` (100% nulo para bairros e sem informações completas de particionamento).
2. **Dados Enriquecidos (Origem de Referência):** `C:/Users/Boanerges/Desktop/Projetos/Report Preview/data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv`
   - Contém bairros preenchidos para ocorrências em Fortaleza.
   - Contém coordenadas em float64.
   - Contém o campo `ais` que apresenta nulos (cerca de 50%), a ser completado via heurística espacial e mapeamento bairro-AIS.
