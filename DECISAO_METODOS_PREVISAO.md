# Decisao Metodologica sobre os Metodos de Previsao

Data de consolidacao: 2026-06-06

## Objetivo deste documento

Este documento registra a justificativa tecnica para os metodos de previsao efetivamente utilizados no projeto nesta fase e explica por que eles foram adotados antes dos pacotes sugeridos pelo orientador, em especial `skforecast`.

## Resumo da decisao

Nesta etapa do projeto, foram priorizados modelos univariados simples, auditaveis e compativeis com a estrutura experimental exigida pelo orientador:

- `naive_lag1`
- `linear_regression`
- `ridge`
- `poisson`

A decisao foi manter o pipeline sob controle direto do projeto enquanto o protocolo experimental era estabilizado:

- selecao das unidades que cobrem 95% dos eventos
- previsao univariada por unidade espacial
- comparacao entre bairros e hexagonos
- calculo de metricas por unidade e globais

O resultado foi suficiente para responder a pergunta principal desta fase: se a divisao hexagonal melhora a previsao em relacao aos bairros.

## Ponto central: `skforecast` nao e um modelo, e sim um framework

O `skforecast` e uma biblioteca para organizar fluxos de previsao de series temporais com estimadores compativeis com a API do scikit-learn. Em outras palavras, ele facilita:

- criacao de lags
- estrategias recursivas e diretas
- backtesting
- tuning
- previsao de serie unica ou multiplas series

Mas ele nao substitui a decisao principal deste projeto, que e escolher:

- qual representacao espacial usar
- qual protocolo experimental usar
- qual estimador estatistico ou de machine learning gera melhor previsao

Por isso, nesta fase, a comparacao metodologica principal continuou sendo entre os estimadores e nao entre frameworks.

## Por que os modelos atuais foram escolhidos

### 1. `naive_lag1`

Foi incluido como baseline minimo.

Justificativa:

- e um comparador simples e interpretavel
- mostra se os modelos mais elaborados realmente agregam valor
- em series esparsas, e comum que um baseline ingenuo seja competitivo em alguns casos

Sem esse baseline, o projeto correria o risco de adotar modelos mais complexos sem demonstrar ganho real.

### 2. `linear_regression`

Foi usada como baseline supervisionado de baixa complexidade.

Justificativa:

- e simples de treinar e interpretar
- permite testar rapidamente o efeito dos lags e das features temporais
- tem baixo custo computacional
- fornece uma referencia clara para comparacoes posteriores

Ela ajuda a verificar se a estrutura de features criada pelo projeto carrega algum sinal preditivo.

### 3. `ridge`

Foi escolhida como extensao natural da regressao linear.

Justificativa:

- reduz instabilidade numerica quando as features de lag sao correlacionadas
- tende a generalizar melhor que a regressao linear pura em alguns cenarios
- mantem boa interpretabilidade e baixo custo de execucao

Como o conjunto de preditores do projeto inclui lags e medidas derivadas, a regularizacao faz sentido como baseline robusto.

### 4. `poisson`

Foi o metodo mais importante desta fase e acabou sendo o vencedor.

Justificativa:

- o alvo do projeto e contagem semanal de eventos
- os dados sao esparsos, com muitas semanas de valor zero
- previsoes negativas nao fazem sentido para esse problema
- a regressao de Poisson e mais coerente com contagens raras do que a regressao linear classica

Essa escolha foi especialmente adequada para o contexto de CVLI semanal por bairro e por hexagono. Os resultados mostraram que o modelo `poisson` venceu tanto no cenario de bairros quanto no de hexagonos.

## Por que esses modelos foram preferidos nesta fase em vez do `skforecast`

### 1. O experimento precisava primeiro de estabilidade metodologica

Antes de trocar a base de modelagem para um framework externo, era necessario fechar o protocolo do orientador:

- corte de cobertura de 95%
- previsao por unidade espacial
- metricas por unidade
- media e variancia globais
- comparacao bairros vs hexagonos

Adotar `skforecast` antes disso aumentaria o numero de variaveis mudando ao mesmo tempo:

- framework
- pipeline de features
- estrategia de previsao
- protocolo experimental

Do ponto de vista cientifico, isso dificultaria a interpretacao do que realmente causou um eventual ganho ou perda de desempenho.

### 2. Os modelos atuais sao mais faceis de auditar

O projeto tem um objetivo academico e precisa de rastreabilidade.

Com o pipeline atual, e facil mostrar:

- como as series semanais foram construidas
- como os lags foram montados
- como cada modelo foi treinado
- como as metricas foram calculadas

Essa transparencia e especialmente importante para defesa metodologica no mestrado.

### 3. O custo de dependencia foi mantido baixo

O projeto atual depende basicamente de:

- `pandas`
- `numpy`
- `scikit-learn`
- `shapely`

Isso reduz risco de incompatibilidade, simplifica reproducao e facilita manutencao do experimento.

Nesta fase, a equipe precisava primeiro provar o protocolo e os resultados, nao ampliar a pilha tecnologica.

### 4. O ganho esperado do `skforecast` nesta fase era mais operacional do que cientifico

Para o problema atual, o `skforecast` agregaria principalmente:

- conveniencia
- padronizacao
- backtesting mais pronto
- suporte mais organizado para multi-step e multi-series

Mas o ganho cientifico imediato para a pergunta principal desta fase nao era obrigatorio, porque:

- ja era possivel gerar previsoes univariadas por unidade
- ja era possivel comparar bairros e hexagonos
- ja era possivel identificar o modelo vencedor

Ou seja, nesta etapa ele seria mais uma melhoria de engenharia do que uma necessidade metodologica.

## Por que `skforecast` continua sendo recomendado para a proxima fase

Embora nao tenha sido adotado como base principal nesta etapa, o `skforecast` continua sendo uma opcao muito forte para as proximas fases.

Razoes:

- suporta workflows recursivos e diretos
- suporta multiplas series
- facilita backtesting formal
- aceita estimadores compativeis com `scikit-learn`
- e adequado para evoluir do univariado para o multivariado

Conclusao pratica:

- nesta fase, o projeto privilegiou simplicidade, auditabilidade e controle experimental
- na proxima fase, o `skforecast` e candidato natural para reorganizar o pipeline sem abandonar os estimadores ja validados

## Sobre o pacote `scikit-forecasts`

O pacote `scikit-forecasts`, tambem citado nas referencias, nao foi priorizado.

Motivos:

- ecossistema muito menos maduro
- menor sinal de manutencao ativa
- documentacao e adocao menos consolidadas
- menor confianca para servir como base principal de um experimento academico reproduzivel

Assim, entre os pacotes sugeridos, `skforecast` e o candidato realmente forte para incorporacao futura.

## Conclusao final

A decisao de usar `naive_lag1`, `linear_regression`, `ridge` e `poisson` nesta fase foi tecnicamente justificavel e coerente com o objetivo do projeto.

Esses modelos foram preferidos nesta etapa porque:

- permitiram fechar o protocolo experimental do orientador com clareza
- sao simples de explicar e defender academicamente
- sao compativeis com dados de contagem semanal esparsa
- reduziram risco de complexidade desnecessaria
- entregaram resultado comparativo suficiente para responder a hipotese principal

Entre eles, o `poisson` foi a escolha mais apropriada para o fenomeno modelado e apresentou o melhor desempenho global.

O `skforecast` nao foi rejeitado; ele foi apenas postergado para a proxima fase, quando seu maior valor aparecera na expansao para:

- previsao operacional da proxima semana
- previsao multi-step
- previsao multivariada ou multi-series
- backtesting mais sofisticado

## Referencias externas consultadas

- PyPI do skforecast:
  [https://pypi.org/project/skforecast/](https://pypi.org/project/skforecast/)
- Documentacao oficial do skforecast:
  [https://skforecast.org/](https://skforecast.org/)
- Classe `ForecasterRecursiveMultiSeries`:
  [https://skforecast.org/latest/api/forecasterrecursivemultiseries.html/](https://skforecast.org/latest/api/forecasterrecursivemultiseries.html/)
- Guia oficial de estrategia direta:
  [https://skforecast.org/latest/user_guides/direct-multi-step-forecasting.html](https://skforecast.org/latest/user_guides/direct-multi-step-forecasting.html)
- PyPI do `scikit-forecasts`:
  [https://pypi.org/project/scikit-forecasts/](https://pypi.org/project/scikit-forecasts/)
