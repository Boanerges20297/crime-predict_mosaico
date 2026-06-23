# Benchmark de Modelos para Séries Semanais por Hexágono (CVLI)

## Ranking dos modelos

| Modelo | MSE | MAE | Hexágonos avaliados |
| :--- | ---: | ---: | ---: |
| poisson | 0.0581 | 0.1403 | 125 |
| ridge | 0.0583 | 0.1397 | 125 |
| linear_regression | 0.0585 | 0.1399 | 125 |
| hist_gradient_boosting | 0.0702 | 0.1466 | 125 |
| random_forest | 0.0871 | 0.1448 | 125 |
| naive_lag1 | 0.0930 | 0.0703 | 125 |

## Melhor modelo

- **Modelo vencedor:** `poisson`
- **Arquivo de previsões por hexágono:** `C:\Users\STI01\Desktop\Projetos\crime-predict_mosaico\data\processed\cvli_hex_forecasts.csv`

Cada previsão é individualizada por hexágono, usando a série semanal histórica daquela célula.