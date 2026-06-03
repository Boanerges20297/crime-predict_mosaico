# Requisitos do Projeto

## Requisitos Funcionais (RF)
1. **RF01 - Carga e Limpeza de Dados:** O sistema deve importar os dados de criminalidade enriquecidos e filtrar apenas registros ocorridos na cidade de Fortaleza.
2. **RF02 - Mapeamento e Imputação de AIS:** O sistema deve preencher a coluna `ais` nula nas ocorrências de Fortaleza utilizando a associação Bairro $\rightarrow$ AIS e cálculo espacial de menor distância para bairros desconhecidos.
3. **RF03 - Geração de Séries Temporais Semanais:** O sistema deve agrupar o número de crimes semanalmente para cada célula espacial (Bairro, AIS e Célula Hexagonal).
4. **RF04 - Predição Temporal (RLM):** O sistema deve treinar e validar modelos de Regressão Linear Múltipla para prever o volume de crimes da próxima semana com base em lags temporais (ex: últimas 3 semanas).
5. **RF05 - Modelagem de Grade Hexagonal Móvel:** O sistema deve gerar grades de hexágonos regulares a partir de 4 parâmetros dinâmicos:
   - Deslocamento horizontal ($dx$)
   - Deslocamento vertical ($dy$)
   - Ângulo de Rotação ($\theta$)
   - Raio do hexágono ($R$)
6. **RF06 - Busca Otimizada por Algoritmo Genético:** O algoritmo evolucionário deve encontrar o conjunto de parâmetros ($dx, dy, \theta, R$) da grade que minimiza o EQM (Erro Quadrático Médio) agregado das previsões da RLM.
7. **RF07 - Relatório Comparativo de Performance:** O sistema deve gerar uma comparação estatística final do EQM entre Bairros, AIS e Mosaico Hexagonal Ótimo.

## Requisitos Não-Funcionais (RNF)
1. **RNF01 - Portabilidade:** O pipeline deve ser executado localmente em ambiente Windows utilizando Python standard e bibliotecas científicas populares (Pandas, Numpy, Scikit-learn, Shapely).
2. **RNF02 - Desempenho:** A geração do grid hexagonal e o cálculo do EQM para a RLM devem ser eficientes para permitir a avaliação de centenas de gerações/indivíduos no AG em tempo hábil.
3. **RNF03 - Reprodutibilidade:** Toda geração de números aleatórios (inicialização da população do AG, mutações, etc.) deve permitir a definição de um `seed` para consistência de resultados.
