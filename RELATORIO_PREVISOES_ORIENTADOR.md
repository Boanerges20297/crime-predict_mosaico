# Relatorio de Previsoes no Protocolo do Orientador

Este relatorio resume as previsoes geradas no protocolo experimental novo, com comparacao entre bairros e hexagonos usando:

- retencao das unidades espaciais que cobrem 95% dos eventos
- previsao univariada por unidade espacial
- modelo vencedor `poisson`
- janela de teste de `2026-01-05` a `2026-05-25`

## 1. O que foi previsto

As previsoes atualmente salvas no protocolo novo sao previsoes de teste, usadas para validacao do modelo no periodo final da serie historica.

- Bairros:
  `data/processed/baseline_bairros_95_predictions.csv`
- Hexagonos:
  `data/processed/experimento_hexagonos_95_predictions.csv`

Isso significa que cada linha compara:

- `y_true`: numero real de eventos na semana
- `y_pred`: numero previsto pelo modelo para a mesma semana

Ainda nao foi gerado, nesse protocolo novo, um arquivo dedicado so com a previsao operacional da proxima semana para cada bairro ou hexagono. O que existe hoje e a validacao preditiva no conjunto de teste.

## 2. Quantidade de previsoes geradas

- Bairros:
  86 bairros retidos no corte de 95%
- Hexagonos:
  94 hexagonos retidos no corte de 95%

Cada unidade teve 21 semanas no conjunto de teste.

- Total de previsoes em bairros:
  1806 linhas
- Total de previsoes em hexagonos:
  1974 linhas

## 3. Desempenho global das previsoes

### Bairros

- Cobertura atingida:
  0.9522
- Modelo vencedor:
  `poisson`
- Media do MSE:
  0.081781
- Variancia do MSE:
  0.015371
- Media do MAE:
  0.187594
- Variancia do MAE:
  0.020896

### Hexagonos

- Cobertura atingida:
  0.9519
- Modelo vencedor:
  `poisson`
- Media do MSE:
  0.077519
- Variancia do MSE:
  0.013910
- Media do MAE:
  0.178733
- Variancia do MAE:
  0.013848

## 4. Interpretacao comparativa

No protocolo novo, as previsoes por hexagono tiveram desempenho global ligeiramente melhor que as previsoes por bairro:

- menor MSE medio
- menor MAE medio
- menor variabilidade entre unidades

Isso reforca a hipotese do orientador de que uma divisao espacial alternativa aos bairros pode melhorar a previsao.

## 5. Unidades com melhor desempenho

### Melhores bairros por MSE

- `CONJUNTO CEARA I`: MSE 0.000688, MAE 0.025472
- `JARDIM GUANABARA`: MSE 0.000778, MAE 0.027685
- `AUTRAN NUNES`: MSE 0.001130, MAE 0.033012
- `JOAQUIM TAVORA`: MSE 0.001296, MAE 0.035763
- `COACU`: MSE 0.001916, MAE 0.043522

### Melhores hexagonos por MSE

- `hex 33`: MSE 0.000564, MAE 0.022751
- `hex 120`: MSE 0.000756, MAE 0.026983
- `hex 112`: MSE 0.001479, MAE 0.038031
- `hex 35`: MSE 0.002510, MAE 0.050079
- `hex 126`: MSE 0.002818, MAE 0.052941

Essas unidades sao as mais estaveis e mais bem ajustadas pelo modelo escolhido.

## 6. Unidades com pior desempenho

### Piores bairros por MSE

- `MESSEJANA`: MSE 0.646350, MAE 0.660944
- `LAGOA REDONDA`: MSE 0.521615, MAE 0.423685
- `JANGURURSSU`: MSE 0.517440, MAE 0.692414
- `JOSE DE ALENCAR`: MSE 0.495752, MAE 0.485294
- `BARRA DO CEARA`: MSE 0.365724, MAE 0.596052

### Piores hexagonos por MSE

- `hex 91`: MSE 0.607830, MAE 0.484320
- `hex 100`: MSE 0.558254, MAE 0.635079
- `hex 81`: MSE 0.546287, MAE 0.496649
- `hex 28`: MSE 0.447635, MAE 0.319755
- `hex 108`: MSE 0.360725, MAE 0.259607

Essas unidades devem concentrar a analise detalhada posterior, porque sao as regioes onde o modelo encontra mais dificuldade.

## 7. Sinais observados nas previsoes

Os arquivos de previsao mostram um comportamento importante:

- muitas unidades apresentam previsoes fracionarias baixas, tipicas de eventos raros por semana
- em varias unidades, o modelo tende a superestimar semanas sem evento
- isso e coerente com a natureza esparsa do CVLI semanal por celula

Exemplos de sobrestimacao acumulada em bairros:

- `JANGURURSSU`: previsao acumulada muito acima do observado no teste
- `BARRA DO CEARA`: previsao acumulada acima do observado
- `GRANJA LISBOA`: previsao acumulada acima do observado

Exemplos de sobrestimacao acumulada em hexagonos:

- `hex 66`
- `hex 50`
- `hex 24`
- `hex 8`

Isso nao invalida o experimento, mas mostra que ainda ha espaco para calibracao, principalmente nas unidades com maior volatilidade.

## 8. O que ja pode ser dito sobre as previsoes

Ja e possivel afirmar que:

- o pipeline de previsao univariada por unidade espacial esta implementado
- as previsoes foram geradas e validadas no conjunto de teste
- os hexagonos tiveram melhor desempenho global que os bairros no protocolo novo
- existe heterogeneidade forte entre unidades, com algumas regioes muito previsiveis e outras bem mais dificeis

## 9. O que ainda falta, se o objetivo for previsao operacional

Se voce precisar de um relatorio orientado a uso final, ainda faltam duas entregas possiveis:

- gerar um arquivo com a previsao da proxima semana para cada bairro no protocolo novo
- gerar um arquivo com a previsao da proxima semana para cada hexagono no protocolo novo

Hoje, o protocolo novo responde muito bem a pergunta cientifica de validacao. Para responder a pergunta operacional "o que o modelo preve para a semana seguinte?", vale criar uma etapa final de treinamento completo e forecast `t+1` por unidade.

## 10. Arquivos principais

- `data/processed/baseline_bairros_95_predictions.csv`
- `data/processed/experimento_hexagonos_95_predictions.csv`
- `data/processed/baseline_bairros_95_metrics.csv`
- `data/processed/experimento_hexagonos_95_metrics.csv`
- `data/processed/baseline_bairros_95_summary.json`
- `data/processed/experimento_hexagonos_95_summary.json`
- `RELATORIO_COMPARATIVO_ORIENTADOR.md`
