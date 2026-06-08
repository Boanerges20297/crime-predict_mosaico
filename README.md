# Busca evolucionaria pelo mosaico hexagonal regular otimo para previsao espaco-temporal de crimes em Fortaleza/CE

Este projeto implementa um Algoritmo Genetico (AG) para encontrar um particionamento hexagonal do territorio de Fortaleza/CE e comparar seu desempenho preditivo com a divisao tradicional por bairros.

## Mapa interativo

O projeto gera um mapa em `mapa_interativo_fortaleza.html` com comparacao entre:
- Areas Integradas de Seguranca (AIS)
- Bairros
- Mosaico Hexagonal Otimo

## Como funciona o AG

Cada individuo usa quatro genes:
- `dx`: deslocamento horizontal inicial
- `dy`: deslocamento vertical inicial
- `theta`: rotacao da grade
- `R`: raio do hexagono

A aptidao e calculada a partir do erro medio de previsao apos:
- gerar a grade hexagonal
- associar crimes aos hexagonos
- agregar contagens por semana
- criar lags temporais
- treinar e avaliar o modelo preditivo

## Execucao do projeto

1. Instale as dependencias:
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
4. Gere o baseline formal por bairros com cobertura de 95%:
   ```bash
   python src/run_baseline_bairros.py
   ```
5. Gere o experimento formal por hexagonos com cobertura de 95%:
   ```bash
   python src/run_hex_experiment.py
   ```
6. Gere a comparacao final no protocolo do orientador:
   ```bash
   python src/run_orientador_comparison.py
   ```
7. Opcional: rode o sweep de tamanhos de malha no protocolo do orientador:
   ```bash
   python src/run_hex_sweep_orientador.py
   ```
8. Rode a fase 2 com `skforecast` em bairros:
   ```bash
   python src/run_skforecast_phase2_bairros.py
   ```
9. Rode a fase 2 com `skforecast` em hexagonos:
   ```bash
   python src/run_skforecast_phase2_hexagonos.py
   ```
10. Gere a comparacao consolidada da fase 2 com `skforecast`:
   ```bash
   python src/run_skforecast_phase2_comparison.py
   ```
11. Gere a avaliacao comparativa legada:
   ```bash
   python src/evaluate_comparison.py
   ```
12. Gere o mapa HTML:
   ```bash
   python src/generate_map.py
   ```

## Aplicacao Flask

Para observar o comportamento da malha em tempo de execucao:

```bash
python app.py
```

Depois abra [http://127.0.0.1:5000](http://127.0.0.1:5000).

A interface permite:
- recalcular a malha hexagonal dinamicamente
- filtrar por um ou mais bairros
- filtrar por intervalo de datas
- ajustar populacao e geracoes do AG
- visualizar a area de estudo inferida e os hexagonos otimos no mapa

## Observacoes atuais

- A base priorizada pela aplicacao e `data/processed/fortaleza_crimes_normalizado.csv`.
- O recorte espacial dos hexagonos usa uma area de estudo inferida a partir dos dados filtrados, evitando o vazamento simples da `bounding box`.
- Para estudos mais rigorosos, o ideal e substituir a area inferida por um limite oficial de Fortaleza em formato geografico.
- O protocolo novo do orientador salva artefatos dedicados em `data/processed/` para bairros e hexagonos com filtro de cobertura de `95%`.
- A fase 2 com `skforecast` salva um segundo conjunto de artefatos com previsoes de validacao e previsao operacional `t+1`.
