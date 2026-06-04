# Busca evolucionária pelo mosaico hexagonal regular ótimo para previsão espaço-temporal de crimes em Fortaleza/CE

Este projeto implementa um **Algoritmo Genético (AG)** para encontrar o particionamento ótimo do território da Cidade de Fortaleza/CE em uma grade de hexágonos regulares parametrizada. O objetivo é encontrar a divisão espacial que minimiza o erro de previsão do número de crimes semanais em cada célula territorial por meio de um modelo preditivo de **Regressão Linear Múltipla (RLM)**.

---

## Mapa interativo

O projeto gera um mapa em `mapa_interativo_fortaleza.html` com comparação entre:
- **Áreas Integradas de Segurança (AIS)**
- **Bairros**
- **Mosaico Hexagonal Ótimo**

---

## Como funciona o AG

Cada indivíduo usa quatro genes:
- `dx`: deslocamento horizontal inicial;
- `dy`: deslocamento vertical inicial;
- `theta`: rotação da grade;
- `R`: raio do hexágono.

A aptidão é calculada a partir do erro quadrático médio de previsão após:
- gerar a grade hexagonal;
- associar crimes aos hexágonos;
- agregar contagens por semana;
- criar lags temporais;
- treinar e avaliar o modelo preditivo.

---

## Execução do projeto

1. Instale as dependências:
   ```bash
   pip install pandas numpy shapely scikit-learn folium flask openpyxl
   ```
2. Processe a base:
   ```bash
   python src/prepare_data.py
   ```
3. Execute o AG:
   ```bash
   python src/run_ga.py
   ```
4. Gere a avaliação comparativa:
   ```bash
   python src/evaluate_comparison.py
   ```
5. Gere o mapa HTML:
   ```bash
   python src/generate_map.py
   ```

---

## Aplicação Flask

Para observar o comportamento da malha em tempo de execução:

```bash
python app.py
```

Depois abra [http://127.0.0.1:5000](http://127.0.0.1:5000).

A interface permite:
- recalcular a malha hexagonal dinamicamente;
- filtrar por um ou mais bairros;
- filtrar por intervalo de datas;
- ajustar população e gerações do AG;
- visualizar a área de estudo inferida e os hexágonos ótimos no mapa.

---

## Observações atuais

- A base priorizada pela aplicação é `data/processed/fortaleza_crimes_normalizado.csv`.
- O recorte espacial dos hexágonos usa uma área de estudo inferida a partir dos dados filtrados, evitando o vazamento simples da `bounding box`.
- Para estudos mais rigorosos, o ideal é substituir a área inferida por um limite oficial de Fortaleza em formato geográfico.
