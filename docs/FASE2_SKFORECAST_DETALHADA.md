# Fase 2 com skforecast: Documentacao Detalhada

Data de consolidacao: 2026-06-06

## 1. Objetivo da fase 2

A fase 2 introduz o `skforecast` no projeto para testar a hipotese de que um framework dedicado a series temporais pode melhorar a qualidade das previsoes sem quebrar a comparabilidade com a fase 1.

Para isso, a fase 2 manteve o mesmo protocolo experimental:

- apenas eventos CVLI
- apenas Fortaleza
- series semanais por unidade espacial
- retencao das unidades que cobrem 95% dos eventos
- comparacao entre bairros e hexagonos
- avaliacao no mesmo corte temporal de treino e teste

A diferenca principal foi a troca do pipeline manual da fase 1 por um forecaster recursivo do `skforecast`.

## 2. Biblioteca adotada

- Biblioteca:
  `skforecast`
- Versao instalada:
  `0.22.0`
- Estrategia utilizada:
  `ForecasterRecursive`

Essa escolha foi coerente com a documentacao oficial do pacote, que posiciona `ForecasterRecursive` como o forecaster padrao para previsao recursiva de serie unica com estimadores compativeis com a API do scikit-learn.

## 3. Desenho metodologico

### 3.1 Estrutura espacial

Dois cenarios foram avaliados:

- bairros
- hexagonos

Os mesmos criterios da fase 1 foram mantidos:

- bairros retidos:
  86
- hexagonos retidos:
  94
- cobertura dos bairros:
  95,22%
- cobertura dos hexagonos:
  95,19%

### 3.2 Estrutura temporal

As series foram agregadas semanalmente.

Para cada unidade espacial:

- periodo de treino:
  ate `2025-12-31`
- periodo de teste:
  de `2026-01-05` ate `2026-05-25`
- horizonte de teste por unidade:
  21 semanas

### 3.3 Estrategia de previsao

Foi utilizada previsao recursiva univariada:

- o modelo aprende a prever a proxima observacao com base em lags
- as previsoes seguintes usam valores previstos anteriormente

Nesta fase, os lags usados foram:

- `1`
- `2`
- `3`

## 4. Modelos avaliados dentro do skforecast

Os estimadores testados foram:

- `linear_regression`
- `ridge`
- `poisson`
- `hist_gradient_boosting`

Todos foram acoplados ao `ForecasterRecursive`.

O objetivo era responder duas perguntas:

1. o `skforecast` melhora ou nao os resultados em relacao a fase 1?
2. o melhor estimador continua sendo o mesmo nos dois recortes espaciais?

## 5. Implementacao no projeto

Arquivos principais criados para a fase 2:

- [src/skforecast_protocol.py](C:/Users/Boanerges/Desktop/Projetos/Crime_Predict-Algoritmo Genetico/src/skforecast_protocol.py)
- [src/run_skforecast_phase2_bairros.py](C:/Users/Boanerges/Desktop/Projetos/Crime_Predict-Algoritmo Genetico/src/run_skforecast_phase2_bairros.py)
- [src/run_skforecast_phase2_hexagonos.py](C:/Users/Boanerges/Desktop/Projetos/Crime_Predict-Algoritmo Genetico/src/run_skforecast_phase2_hexagonos.py)
- [src/run_skforecast_phase2_comparison.py](C:/Users/Boanerges/Desktop/Projetos/Crime_Predict-Algoritmo Genetico/src/run_skforecast_phase2_comparison.py)

Dependencias atualizadas:

- [requirements.txt](C:/Users/Boanerges/Desktop/Projetos/Crime_Predict-Algoritmo Genetico/requirements.txt)

## 6. Artefatos gerados

### 6.1 Bairros

- `data/processed/fase2_skforecast_bairros_95_weekly_series.csv`
- `data/processed/fase2_skforecast_bairros_95_coverage.csv`
- `data/processed/fase2_skforecast_bairros_95_metrics.csv`
- `data/processed/fase2_skforecast_bairros_95_predictions.csv`
- `data/processed/fase2_skforecast_bairros_95_next_step_forecasts.csv`
- `data/processed/fase2_skforecast_bairros_95_summary.json`
- [FASE2_SKFORECAST_BAIRROS_95.md](C:/Users/Boanerges/Desktop/Projetos/Crime_Predict-Algoritmo Genetico/FASE2_SKFORECAST_BAIRROS_95.md)

### 6.2 Hexagonos

- `data/processed/fase2_skforecast_hexagonos_95_weekly_series.csv`
- `data/processed/fase2_skforecast_hexagonos_95_coverage.csv`
- `data/processed/fase2_skforecast_hexagonos_95_metrics.csv`
- `data/processed/fase2_skforecast_hexagonos_95_predictions.csv`
- `data/processed/fase2_skforecast_hexagonos_95_next_step_forecasts.csv`
- `data/processed/fase2_skforecast_hexagonos_95_summary.json`
- [FASE2_SKFORECAST_HEXAGONOS_95.md](C:/Users/Boanerges/Desktop/Projetos/Crime_Predict-Algoritmo Genetico/FASE2_SKFORECAST_HEXAGONOS_95.md)

### 6.3 Comparacao consolidada

- `data/processed/fase2_skforecast_comparison.json`
- [RELATORIO_FASE2_SKFORECAST.md](C:/Users/Boanerges/Desktop/Projetos/Crime_Predict-Algoritmo Genetico/RELATORIO_FASE2_SKFORECAST.md)

## 7. Resultados encontrados

### 7.1 Bairros

Melhor modelo:

- `poisson`

Metricas globais:

- media do MSE:
  `0.082041`
- variancia do MSE:
  `0.014875`
- media do MAE:
  `0.187150`
- variancia do MAE:
  `0.021579`

### 7.2 Hexagonos

Melhor modelo:

- `hist_gradient_boosting`

Metricas globais:

- media do MSE:
  `0.074178`
- variancia do MSE:
  `0.013288`
- media do MAE:
  `0.170417`
- variancia do MAE:
  `0.012171`

### 7.3 Melhor cenario global

O melhor cenario global da fase 2 foi:

- `fase2_skforecast_hexagonos_95`

Isso significa que:

- a malha hexagonal continuou superior aos bairros
- com `skforecast`, o ganho dos hexagonos ficou ainda mais claro
- no recorte hexagonal, um modelo nao linear (`hist_gradient_boosting`) superou o `poisson`

## 8. Comparacao com a fase 1

### 8.1 Bairros

Fase 1:

- MSE medio:
  `0.081781`
- MAE medio:
  `0.187594`
- melhor modelo:
  `poisson`

Fase 2:

- MSE medio:
  `0.082041`
- MAE medio:
  `0.187150`
- melhor modelo:
  `poisson`

Leitura:

- o desempenho em bairros ficou praticamente estavel
- o `skforecast` nao trouxe ganho material para bairros nessa configuracao
- o `poisson` continuou sendo a melhor escolha nesse nivel espacial

### 8.2 Hexagonos

Fase 1:

- MSE medio:
  `0.077519`
- MAE medio:
  `0.178733`
- melhor modelo:
  `poisson`

Fase 2:

- MSE medio:
  `0.074178`
- MAE medio:
  `0.170417`
- melhor modelo:
  `hist_gradient_boosting`

Leitura:

- houve melhora real com `skforecast` no recorte hexagonal
- o melhor modelo deixou de ser linear/Poisson e passou a ser um estimador de boosting
- isso sugere que a malha hexagonal concentra melhor estruturas locais que um modelo nao linear consegue explorar

## 9. Previsao operacional t+1

Diferente da fase 1, a fase 2 tambem gerou previsoes operacionais para a proxima semana apos a ultima observacao disponivel.

### 9.1 Bairros com maior previsao t+1

Principais bairros previstos:

- `JANGURURSSU`: `0.617982`
- `BARRA DO CEARA`: `0.552151`
- `MESSEJANA`: `0.488943`
- `GRANJA LISBOA`: `0.367894`
- `MONDUBIM`: `0.352032`

Arquivo:

- `data/processed/fase2_skforecast_bairros_95_next_step_forecasts.csv`

### 9.2 Hexagonos com maior previsao t+1

Principais hexagonos previstos:

- `hex 24`: `0.627379`
- `hex 100`: `0.497988`
- `hex 66`: `0.322504`
- `hex 50`: `0.306172`
- `hex 14`: `0.300903`

Arquivo:

- `data/processed/fase2_skforecast_hexagonos_95_next_step_forecasts.csv`

Essas previsoes sao valores esperados de contagem semanal, nao probabilidades diretas de ocorrencia.

## 10. Interpretacao tecnica

### 10.1 O que a fase 2 mostrou

A fase 2 mostrou tres pontos importantes:

1. o `skforecast` e viavel no projeto sem alterar o protocolo central
2. bairros continuam melhor ajustados por um modelo de contagem simples
3. hexagonos se beneficiam mais de um modelo recursivo com maior capacidade nao linear

### 10.2 O que isso significa para a pesquisa

Do ponto de vista do mestrado, a fase 2 fortalece a narrativa cientifica:

- a superioridade dos hexagonos nao depende apenas do pipeline manual da fase 1
- quando a modelagem e refinada com um framework dedicado, os hexagonos permanecem melhores
- o melhor metodo pode depender da representacao espacial usada

Esse ultimo ponto e particularmente relevante:

- em bairros, `poisson` continuou adequado
- em hexagonos, `hist_gradient_boosting` apareceu como melhor alternativa

## 11. Limitacoes da fase 2

- o forecaster usado foi univariado por unidade espacial
- ainda nao foi usada modelagem multiseries global
- ainda nao foram incluidos exogenos mais ricos
- os avisos do `skforecast` sobre bins duplicados indicam forte repeticao de valores previstos em series esparsas, o que e coerente com o problema mas sugere cautela em extensoes probabilisticas

## 12. Conclusao final da fase 2

A fase 2 com `skforecast` foi concluida com sucesso e agregou valor ao projeto.

Conclusoes centrais:

- `skforecast` foi integrado ao pipeline mantendo comparabilidade com a fase 1
- bairros permaneceram com melhor ajuste via `poisson`
- hexagonos melhoraram com `hist_gradient_boosting`
- o melhor desempenho global da fase 2 ficou com os hexagonos
- a fase 2 ja entrega previsao de validacao e previsao operacional `t+1`

Em termos estrategicos, a fase 2 valida o `skforecast` como base adequada para a proxima expansao do projeto:

- previsao multivariada
- forecasters multiseries
- backtesting mais sofisticado
- tuning sistematico de hiperparametros e lags
