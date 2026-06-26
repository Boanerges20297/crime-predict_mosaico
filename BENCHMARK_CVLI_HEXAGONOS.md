# Benchmark de Modelos para Séries Semanais por Hexágono (CVLI)

## Ranking dos modelos

| Modelo | MSE | MAE | Hexágonos avaliados |
| :--- | ---: | ---: | ---: |
| ridge | 0.0361 | 0.1018 | 176 |
| poisson | 0.0362 | 0.1021 | 176 |
| linear_regression | 0.0362 | 0.1020 | 176 |
| hist_gradient_boosting | 0.0437 | 0.1059 | 176 |
| random_forest | 0.0533 | 0.1004 | 176 |
| naive_lag1 | 0.0585 | 0.0504 | 176 |

## Melhor modelo

- **Modelo vencedor:** `ridge`
- **Arquivo de previsões por hexágono:** `C:\Users\STI01\Desktop\Projetos\crime-predict_mosaico\data\processed\cvli_hex_forecasts.csv`

Cada previsão é individualizada por hexágono, usando a série semanal histórica daquela célula.