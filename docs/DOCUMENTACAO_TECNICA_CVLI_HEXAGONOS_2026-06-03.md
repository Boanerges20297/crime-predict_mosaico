# Documentação Técnica Consolidada — CVLI, Malha Hexagonal Ótima e Previsão Semanal

- **Data:** 2026-06-03
- **Projeto:** `Crime_Predict-Algoritmo Genetico`
- **Escopo desta documentação:** consolidar o estado atual do projeto até esta data, incluindo preparação dos dados, recorte espacial, formulação matemática da malha hexagonal, benchmark de modelos, séries temporais semanais por hexágono, previsões individualizadas e comportamento do dashboard.

---

## 1. Objetivo atual do sistema

O sistema foi ajustado para:

1. trabalhar **apenas com eventos CVLI**;
2. usar uma **malha hexagonal otimizada por algoritmo genético**;
3. gerar **séries temporais semanais por hexágono**;
4. comparar modelos preditivos para encontrar o mais aderente a essa arquitetura;
5. produzir **previsão individualizada por hexágono**;
6. exibir isso em um **dashboard Flask** com filtros por bairro, período e exibição opcional de células esparsas.

Em resumo: o projeto saiu de uma comparação genérica entre agrupamentos e passou a operar como um fluxo espacial-temporal voltado a **CVLI por célula hexagonal**.

---

## 2. Preparação e normalização dos dados

### 2.1 Base processada preferencial

O sistema atualmente tenta ler, nesta ordem:

1. `C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\data\processed\fortaleza_crimes_normalizado.csv`
2. `C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\data\processed\fortaleza_crimes.csv`

Isso foi centralizado em `C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\src\spatial_utils.py`.

### 2.2 Normalização de bairros

A coluna `bairro` foi padronizada para:

- caixa alta;
- remoção de acentos;
- remoção de espaços redundantes;
- conversão de valores vazios ou `"NULL"` para nulo.

Formulação textual:

\[
\text{bairro\_normalizado} = \operatorname{collapse\_spaces}\left(
\operatorname{remove\_acentos}\left(
\operatorname{upper}\left(
\operatorname{strip}(bairro)
\right)\right)\right)
\]

Exemplos:

- `Joaquim Távora` \(\rightarrow\) `JOAQUIM TAVORA`
- `Genibaú` \(\rightarrow\) `GENIBAU`

Implementação em `C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\src\prepare_data.py`.

### 2.3 Restrição territorial inicial

No pipeline de preparação, os registros foram filtrados para Fortaleza com apoio de:

- coluna `cidade`;
- AIS da capital;
- exclusão de bairros sabidamente fora do município;
- preenchimento de AIS faltante com mapeamento `bairro -> AIS`;
- fallback por vizinho mais próximo quando necessário.

---

## 3. Extração exclusiva de CVLI

O sistema foi restringido para CVLI em `C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\src\spatial_utils.py`.

### 3.1 Regra principal

Se existir coluna `tipo`, o filtro é:

\[
\text{manter} \iff \operatorname{upper}(tipo) = \text{"CVLI"}
\]

### 3.2 Regra de fallback

Se `tipo` não estiver disponível, o sistema procura equivalentes textuais em colunas como `tipo_evento` e `nature`, incluindo padrões como:

- `HOMICIDIO DOLOSO`
- `FEMINICIDIO`
- `LATROCINIO`
- `LESAO CORPORAL SEGUIDA DE MORTE`

---

## 4. Recorte espacial de Fortaleza

### 4.1 Bounding box inicial

Antes de qualquer inferência geométrica, o sistema restringe os pontos a:

- latitude em `[-3.9, -3.6]`
- longitude em `[-38.7, -38.4]`

Isso funciona como um filtro espacial bruto.

### 4.2 Remoção de vazamentos espaciais por cluster principal

Para evitar que poucos pontos periféricos “puxem” a malha para fora de Fortaleza, foi adicionado um filtro por **maior cluster espacial** com `DBSCAN`.

Parâmetros atuais:

- `eps = 0.015`
- `min_samples = 40`

Sejam os pontos espaciais:

\[
P = \{(lon_i, lat_i)\}_{i=1}^{n}
\]

O `DBSCAN` particiona \(P\) em rótulos \(c_i\). O sistema mantém apenas o rótulo válido mais frequente:

\[
c^* = \arg\max_c \#\{i : c_i = c,\; c \neq -1\}
\]

e filtra:

\[
P' = \{(lon_i, lat_i)\in P : c_i = c^*\}
\]

Isso remove ilhas desconectadas e reduz o vazamento da malha em direção a municípios vizinhos.

### 4.3 Área de estudo inferida

Após o filtro do cluster principal, a área de estudo é inferida a partir dos próprios pontos de CVLI.

Fluxo atual:

1. construir `MultiPoint`;
2. tentar `concave_hull` com `concavity/ratio = 0.12`;
3. se falhar, usar `convex_hull`;
4. manter apenas o **maior polígono**;
5. aplicar buffer:

\[
b = \max(\text{span} \cdot 0.008,\; 0.001)
\]

onde:

\[
\text{span} = \max(max\_lon - min\_lon,\; max\_lat - min\_lat)
\]

Resultado:

\[
A_{estudo} = \operatorname{largest\_polygon}(\operatorname{buffer}(H, b))
\]

com \(H\) sendo o casco côncavo/convexo resultante.

### 4.4 Limitação conceitual

O recorte atual é **inferido pelos dados**, não por um shapefile/GeoJSON oficial do município. Portanto, ele é bem melhor que uma simples `bbox`, mas ainda não equivale a um limite cartográfico oficial de Fortaleza.

---

## 5. Formulação geométrica da malha hexagonal

Cada indivíduo do algoritmo genético é dado por:

\[
g = (dx, dy, \theta, R)
\]

onde:

- \(dx\): deslocamento horizontal relativo da malha;
- \(dy\): deslocamento vertical relativo da malha;
- \(\theta\): rotação da malha;
- \(R\): raio do hexágono regular.

### 5.1 Geometria básica do hexágono

Para um hexágono regular:

- largura:

\[
w = 2R
\]

- altura:

\[
h = \sqrt{3}R
\]

- passo horizontal entre centros:

\[
\Delta x = 1.5R
\]

- passo vertical entre linhas:

\[
\Delta y = \sqrt{3}R
\]

Nas colunas alternadas, há um deslocamento adicional de:

\[
\frac{\Delta y}{2}
\]

para garantir o empacotamento hexagonal.

### 5.2 Área de um hexágono regular

A área de cada hexágono é:

\[
A_{hex} = \frac{3\sqrt{3}}{2}R^2
\]

Logo, ao aumentar \(R\), cresce quadraticamente a área de cada célula e tende a cair o número de hexágonos criados.

### 5.3 Rotação da malha

Cada vértice é rotacionado em torno do centróide da área de estudo \((c_x, c_y)\):

\[
x' = \cos(\theta)(x-c_x) - \sin(\theta)(y-c_y) + c_x
\]

\[
y' = \sin(\theta)(x-c_x) + \cos(\theta)(y-c_y) + c_y
\]

### 5.4 Recorte geométrico

Os hexágonos gerados sobre a extensão espacial são recortados pela área de estudo:

\[
H_j^{clip} = H_j \cap A_{estudo}
\]

Somente os polígonos não vazios são mantidos para visualização.

---

## 6. Algoritmo genético para otimização da malha

### 6.1 Espaço de busca

Os limites atuais do AG são:

- \(dx \in [0, 1]\)
- \(dy \in [0, 1]\)
- \(\theta \in [0, \pi/3]\)
- \(R \in [0.0075, 0.03]\)

### 6.2 Hiperparâmetros do AG

No núcleo genético (`src/genetic_algorithm.py`), os defaults são:

- população: `10`
- gerações: `5`
- taxa de mutação: `0.15`
- taxa de crossover: `0.8`
- semente: `42`
- alvo default do núcleo: `250`
- peso default da penalização por quantidade de hexágonos: `6.0`

No dashboard, o recálculo dinâmico usa hoje:

- `DEFAULT_POP_SIZE = 6`
- `DEFAULT_GENERATIONS = 5`
- `DEFAULT_TARGET_HEX_COUNT = 180`
- `DEFAULT_HEX_PENALTY_WEIGHT = 5.5`

### 6.3 Avaliação de um indivíduo

Dado um indivíduo \(g\):

1. gera-se a malha;
2. cada ocorrência é atribuída a um hexágono:

\[
H(p_i) = h_j
\]

3. monta-se a série semanal por hexágono;
4. calcula-se o erro preditivo semanal;
5. penaliza-se o desvio entre o número criado de hexágonos e o alvo desejado.

### 6.4 Erro preditivo

Sejam \(y_k\) os valores reais e \(\hat{y}_k\) as previsões:

\[
EQM = \frac{1}{m}\sum_{k=1}^{m}(y_k - \hat{y}_k)^2
\]

### 6.5 Penalização por quantidade de hexágonos

Se \(N_{hex}\) é o número de hexágonos criados e \(N_{alvo}\) é o alvo:

\[
\delta_{hex} = \frac{|N_{hex} - N_{alvo}|}{\max(N_{alvo}, 1)}
\]

\[
EQM_{pen} = EQM + \lambda \cdot \delta_{hex}
\]

onde \(\lambda\) é `hex_penalty_weight`.

### 6.6 Função de aptidão

\[
fitness(g) = \frac{1}{EQM_{pen}(g) + 10^{-8}}
\]

O melhor indivíduo é aquele que maximiza `fitness`, equivalendo a minimizar \(EQM_{pen}\).

### 6.7 Seleção, cruzamento e mutação

- **Seleção:** torneio com 3 candidatos;
- **Crossover:** combinação linear com \(\alpha \in [0.1, 0.9]\);
- **Mutação:** ruído gaussiano decrescente ao longo das gerações.

Força de mutação:

\[
\sigma_g = 0.1\left(1 - \frac{g}{G}\right)
\]

onde:

- \(g\) = geração corrente;
- \(G\) = número total de gerações.

---

## 7. Quantidade de hexágonos: interpretação correta

O sistema hoje distingue:

### 7.1 Hexágonos criados

\[
N_{criados}
\]

Total de polígonos gerados e mantidos após o recorte com a área de estudo.

### 7.2 Hexágonos renderizados

\[
N_{renderizados}
\]

Total de geometrias disponíveis para renderização no mapa.

### 7.3 Hexágonos ativos

\[
N_{ativos}
\]

Número de hexágonos que realmente receberam ao menos uma ocorrência:

\[
N_{ativos} = \#\{h_j : \exists i,\; H(p_i)=h_j\}
\]

Em geral:

\[
N_{ativos} \leq N_{renderizados} \leq N_{criados}
\]

---

## 8. Séries temporais semanais por hexágono

### 8.1 Agregação semanal

Cada hexágono \(h_j\) gera uma série:

\[
Y_j = \{y_{j,t}\}_{t=1}^{T}
\]

onde \(y_{j,t}\) é a contagem de CVLI na semana \(t\).

### 8.2 Variáveis explicativas atuais

Para cada observação semanal, o sistema cria:

- `lag_1`
- `lag_2`
- `lag_3`
- `media_movel_4`
- `tendencia_1`
- `semana_ano`
- `mes`

Formalmente:

\[
lag_k(t) = y_{t-k}
\]

\[
media\_movel\_4(t) = \frac{1}{4}\sum_{r=1}^{4} y_{t-r}
\]

\[
tendencia\_1(t) = y_{t-1} - y_{t-2}
\]

Essas features alimentam tanto o benchmark quanto as previsões finais por hexágono.

### 8.3 Janela temporal atualmente observada

Com a malha final escolhida, a base semanal consolidada possui:

- `28.500` linhas
- `125` hexágonos com série utilizável
- início em `2022-01-17`
- fim em `2026-05-25`

Arquivo:

- `C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\data\processed\cvli_hex_weekly_series.csv`

---

## 9. Benchmark de modelos

### 9.1 Modelos comparados

O benchmark completo atual avaliou:

- `naive_lag1`
- `linear_regression`
- `ridge`
- `poisson`
- `random_forest`
- `hist_gradient_boosting`

### 9.2 Resultado do benchmark final

Ranking atual:

| Modelo | MSE | MAE | Hexágonos avaliados |
| :--- | ---: | ---: | ---: |
| poisson | 0.059982 | 0.142014 | 125 |
| ridge | 0.060116 | 0.141004 | 125 |
| linear_regression | 0.060263 | 0.141177 | 125 |
| hist_gradient_boosting | 0.072816 | 0.149492 | 125 |
| random_forest | 0.090283 | 0.147888 | 125 |
| naive_lag1 | 0.098286 | 0.073143 | 125 |

### 9.3 Interpretação

O melhor modelo foi `poisson`, o que faz bastante sentido para esse problema, porque:

1. a variável-alvo é uma **contagem não negativa**;
2. CVLI por célula semanal é relativamente **esparso**;
3. a estrutura se aproxima mais de um processo de contagem do que de uma variável contínua com ruído gaussiano clássico.

---

## 10. Sweep de quantidade de hexágonos

### 10.1 Motivação

Foi levantada a hipótese de aumentar o raio dos hexágonos para reduzir a fragmentação espacial e melhorar a densidade de informação por célula.

### 10.2 Sweep executado

Foram testadas malhas-alvo:

\[
\{120,\;140,\;160,\;180,\;200\}
\]

com benchmark rápido usando:

- `naive_lag1`
- `linear_regression`
- `ridge`
- `poisson`

### 10.3 Resultado consolidado do sweep

| Alvo | Hex criados | Hex ativos | Melhor modelo rápido | MSE | MAE | Média CVLI/hex | Mediana CVLI/hex | Média atividade | Mediana atividade |
| ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 120 | 119 | 109 | poisson | 0.0656 | 0.1619 | 28.39 | 20.00 | 0.1089 | 0.0866 |
| 140 | 133 | 125 | poisson | 0.0600 | 0.1420 | 24.76 | 18.00 | 0.0955 | 0.0779 |
| 160 | 133 | 125 | poisson | 0.0600 | 0.1420 | 24.76 | 18.00 | 0.0955 | 0.0779 |
| 180 | 133 | 125 | poisson | 0.0600 | 0.1420 | 24.76 | 18.00 | 0.0955 | 0.0779 |
| 200 | 133 | 125 | poisson | 0.0600 | 0.1420 | 24.76 | 18.00 | 0.0955 | 0.0779 |

### 10.4 Malha escolhida

O melhor equilíbrio encontrado foi:

- alvo solicitado: `140`
- hexágonos criados: `133`
- hexágonos ativos: `125`

Essa solução foi persistida em:

- `C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\data\processed\best_hex_grid.json`

Conteúdo atual:

- `dx = 0.8324426408004217`
- `dy = 0.21233911067827616`
- `theta = 0.19040666040567744`
- `R = 0.00935615945025577`
- `best_mse = 0.053355945323510956`
- `penalized_mse = 0.328355945323511`
- `hex_count = 133`
- `active_hex_count = 125`
- `target_hex_count = 140`

### 10.5 Interpretação substantiva

Apesar de termos cogitado ~180 ou ~250 hexágonos, os dados de CVLI empurraram a solução para uma malha menor. Isso é coerente com a estocasticidade do fenômeno: células demais diluem os eventos e pioram a capacidade de aprendizado.

---

## 11. Previsão individualizada por hexágono

Após o benchmark, o melhor modelo é treinado por hexágono usando a série semanal daquela própria célula.

Seja \(h_j\) um hexágono. O modelo aprende:

\[
\hat{y}_{j,t+1} = f_j(x_{j,t})
\]

onde \(x_{j,t}\) contém os lags e atributos temporais de \(h_j\).

Assim, a previsão é **individualizada**:

\[
\hat{Y}_{t+1} = \{\hat{y}_{1,t+1}, \hat{y}_{2,t+1}, \dots, \hat{y}_{J,t+1}\}
\]

e não uma previsão única para a cidade inteira.

### 11.1 Arquivo de previsões

Arquivo gerado:

- `C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\data\processed\cvli_hex_forecasts.csv`

Resumo atual:

- `125` linhas de previsão
- previsão mínima: `0.0008839609840084`
- previsão máxima: `0.4620458620553542`
- previsão média: `0.10243668659501719`

### 11.2 Campos principais por hexágono

Cada linha contém, no mínimo:

- `hex_id`
- `ultima_semana_observada`
- `ultima_contagem`
- `previsao_proxima_semana`
- `media_movel_4`
- `tendencia_1`
- `modelo_previsao`

---

## 12. Dashboard Flask

### 12.1 Estrutura geral

O app principal está em:

- `C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\flask_app.py`

O serviço do dashboard está em:

- `C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\src\dashboard_service.py`

O template principal está em:

- `C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\templates\index.html`

### 12.2 Filtros disponíveis

O dashboard suporta:

- múltiplos bairros;
- data inicial;
- data final;
- população do AG;
- gerações do AG;
- checkbox para ocultar hexágonos esparsos.

### 12.3 Regra de esparsidade

Uma célula é marcada como esparsa se:

\[
\text{activity\_ratio} < 0.35 \cdot \operatorname{median}(\text{activity\_ratio})
\]

ou:

\[
\text{total\_cvli} \leq 2
\]

onde:

\[
\text{activity\_ratio} = \frac{\text{semanas ativas}}{\text{semanas totais}}
\]

### 12.4 Métricas hoje exibidas

O dashboard mostra, entre outras:

- ocorrências filtradas;
- hexágonos criados;
- hexágonos ativos;
- hexágonos esparsos;
- bairros cobertos;
- tempo de execução;
- EQM hexagonal;
- EQM por bairros;
- EQM por AIS;
- modelo semanal por hexágono.

---

## 13. Tooltips e popups analíticos

Os tooltips/popups foram enriquecidos para uso acadêmico e discussão com o orientador.

Hoje, no dashboard, cada hexágono pode exibir:

- identificador do hexágono;
- CVLI total;
- semanas ativas / total;
- taxa de atividade;
- média de CVLI por semana ativa;
- última semana observada;
- última contagem semanal;
- média móvel de 4 semanas;
- tendência recente;
- previsão da próxima semana;
- modelo preditivo utilizado;
- indicador de esparsidade;
- total de hexágonos criados na malha.

Isso foi implementado para:

- dashboard dinâmico;
- mapa HTML estático.

---

## 14. Mapa estático

O script de geração do mapa está em:

- `C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\src\generate_map.py`

Saída gerada:

- `C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\mapa_interativo_fortaleza.html`

O mapa estático agora:

- usa a base normalizada;
- usa apenas CVLI;
- usa a malha hexagonal salva como melhor configuração;
- inclui previsão semanal por hexágono no popup.

---

## 15. Artefatos produzidos até aqui

### 15.1 Documentos

- `C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\HEXAGONOS_OTIMOS_DINAMICA.md`
- `C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\BENCHMARK_CVLI_HEXAGONOS.md`
- `C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\SWEEP_HEXAGONOS_CVLI.md`
- `C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\DOCUMENTACAO_TECNICA_CVLI_HEXAGONOS_2026-06-03.md`

### 15.2 Dados processados

- `C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\data\processed\fortaleza_crimes_normalizado.csv`
- `C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\data\processed\best_hex_grid.json`
- `C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\data\processed\benchmark_cvli_hex_models.json`
- `C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\data\processed\cvli_hex_weekly_series.csv`
- `C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\data\processed\cvli_hex_forecasts.csv`
- `C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\data\processed\cvli_hex_sweep.json`

---

## 16. Ponto importante de consistência atual

Há um detalhe importante no comportamento atual:

- a **malha salva final** em `best_hex_grid.json` está calibrada com alvo `140`;
- o **dashboard sem filtros** reaproveita essa malha salva;
- o **dashboard com filtros ativos**, porém, ainda recalcula usando `DEFAULT_TARGET_HEX_COUNT = 180`.

Isso não invalida o sistema, mas significa que hoje coexistem dois comportamentos:

1. visão padrão de Fortaleza inteira ancorada na melhor malha final do sweep (`140 -> 133`);
2. visão filtrada recalculada com um alvo mais alto (`180`).

Do ponto de vista metodológico, o ideal numa próxima iteração é alinhar explicitamente esses dois regimes.

---

## 17. Limitações atuais

1. **Limite territorial inferido:** ainda não usa polígono oficial de Fortaleza.
2. **Séries esparsas:** CVLI é um fenômeno raro; algumas células tendem a permanecer com baixa densidade.
3. **Dependência do recorte filtrado:** a malha ótima muda com os filtros, então comparações entre recortes precisam ser interpretadas com cautela.
4. **Modelo uniperíodo:** a previsão atual é da próxima semana; ainda não há horizonte multi-step.
5. **Coerência do alvo da malha:** ainda há diferença entre o sweep salvo e o alvo padrão do dashboard filtrado.

---

## 18. Conclusão técnica

Até `2026-06-03`, o projeto está em um estágio em que:

- a base foi normalizada;
- o escopo foi restringido corretamente a CVLI;
- a malha hexagonal deixou de depender apenas de `bbox`;
- o recorte espacial foi melhorado com cluster principal + área inferida;
- houve benchmark formal de modelos para séries semanais por hexágono;
- `poisson` foi o melhor modelo;
- a malha final escolhida ficou em `133` hexágonos criados e `125` ativos;
- as previsões já são individualizadas por hexágono;
- o dashboard e o mapa estático já exibem informações analíticas e preditivas úteis para discussão acadêmica.

Em outras palavras, o sistema já não é apenas um visualizador espacial: ele passou a operar como um **mecanismo preditivo espaço-temporal de CVLI por tesselação hexagonal otimizada**.
