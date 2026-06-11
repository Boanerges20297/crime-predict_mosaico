# Dinamicidade e Formulação Matemática dos Hexágonos Ótimos

## Objetivo

O sistema recalcula uma malha hexagonal ótima para o subconjunto espacial-temporal atualmente filtrado no dashboard. Isso significa que os hexágonos **não são fixos**: eles mudam conforme:

- bairros selecionados;
- intervalo de datas;
- parâmetros do algoritmo genético.

Quando não há filtros, o sistema usa Fortaleza inteira.

---

## Princípio da dinamicidade

Seja o conjunto filtrado de ocorrências:

\[
D_f = \{(lon_i, lat_i, t_i)\}_{i=1}^{n}
\]

onde:

- \(lon_i\) = longitude da ocorrência \(i\)
- \(lat_i\) = latitude da ocorrência \(i\)
- \(t_i\) = data/hora da ocorrência \(i\)

Os filtros definem um subconjunto:

\[
D_f = D \cap B \cap T
\]

em que:

- \(B\) representa o filtro por bairros;
- \(T\) representa o filtro temporal.

A partir de \(D_f\), o sistema:

1. estima a área principal de estudo;
2. calcula a bounding box do subconjunto;
3. executa o AG para encontrar a malha hexagonal ótima;
4. renderiza apenas os hexágonos recortados dentro da área de estudo.

---

## Parâmetros geométricos do hexágono

Cada indivíduo do AG é um vetor:

\[
g = (dx, dy, \theta, R)
\]

onde:

- \(dx\) = deslocamento horizontal inicial;
- \(dy\) = deslocamento vertical inicial;
- \(\theta\) = rotação da malha;
- \(R\) = raio do hexágono regular.

### Geometria básica

Para um hexágono regular:

- largura:
\[
w = 2R
\]

- altura:
\[
h = \sqrt{3}R
\]

- passo horizontal entre colunas:
\[
\Delta x = 0.75 \cdot w = 1.5R
\]

- passo vertical entre linhas:
\[
\Delta y = h = \sqrt{3}R
\]

Cada coluna ímpar recebe um deslocamento vertical adicional de:

\[
\frac{h}{2}
\]

para garantir o empacotamento hexagonal.

---

## Rotação

Cada vértice do hexágono sofre rotação em torno do centro da área de estudo:

\[
x' = \cos(\theta)(x-c_x) - \sin(\theta)(y-c_y) + c_x
\]

\[
y' = \sin(\theta)(x-c_x) + \cos(\theta)(y-c_y) + c_y
\]

onde \((c_x, c_y)\) é o centróide da área usada para a malha.

---

## Função objetivo

A malha é otimizada para melhorar a previsão semanal de crimes.

Depois que os pontos são atribuídos aos hexágonos:

\[
H(p_i) = h_j
\]

o sistema cria séries semanais por hexágono e monta variáveis defasadas:

\[
lag_1, lag_2, lag_3
\]

O modelo preditivo gera um erro quadrático médio:

\[
EQM = \frac{1}{m} \sum_{k=1}^{m} (y_k - \hat{y}_k)^2
\]

A aptidão do AG é:

\[
fitness(g) = \frac{1}{EQM(g) + 10^{-8}}
\]

Logo, o melhor indivíduo é aquele que minimiza o EQM e maximiza a aptidão.

---

## Por que o hexágono ótimo muda?

O hexágono ótimo depende do conjunto filtrado \(D_f\). Ao trocar bairros ou datas:

- mudam os pontos disponíveis;
- muda a área principal de estudo;
- mudam a bounding box e o centróide;
- muda a distribuição espaço-temporal;
- muda o melhor valor de \(R\), \(dx\), \(dy\) e \(\theta\).

Formalmente:

\[
g^* = \arg\max_g fitness(g \mid D_f)
\]

Como \(D_f\) muda com os filtros, o ótimo \(g^*\) também muda.

---

## Quantidade de hexágonos

O sistema detalha três quantidades:

### 1. Hexágonos criados

É o total de polígonos gerados pela malha após o recorte espacial:

\[
N_{criados}
\]

### 2. Hexágonos renderizados

É o total de geometrias efetivamente disponíveis para visualização após interseção com a área de estudo:

\[
N_{renderizados}
\]

### 3. Hexágonos ativos

É o número de hexágonos que realmente receberam ao menos uma ocorrência filtrada:

\[
N_{ativos}
\]

com:

\[
N_{ativos} \leq N_{renderizados} \leq N_{criados}
\]

No dashboard essas três quantidades aparecem explicitamente.

---

## Interpretação prática

- **Criados**: tamanho estrutural da malha.
- **Renderizados**: o que sobra depois do recorte territorial.
- **Ativos**: o subconjunto que realmente participa da análise preditiva.

Se o usuário restringe a análise a poucos bairros ou a uma janela temporal curta, o número de hexágonos ativos tende a cair, e o AG pode preferir valores diferentes para \(R\), \(dx\), \(dy\) e \(\theta\).

---

## Conclusão

O sistema é dinâmico porque a malha ótima é condicionada ao subconjunto filtrado dos dados. Em outras palavras, o dashboard não apenas “esconde” hexágonos: ele recalcula a melhor tesselação para o recorte atual de Fortaleza.
