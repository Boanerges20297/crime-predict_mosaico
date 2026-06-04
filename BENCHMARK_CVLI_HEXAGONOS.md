# Benchmark de Modelos para Séries Semanais por Hexágono (CVLI)

## Ranking dos modelos

| Modelo | MSE | MAE | Hexágonos avaliados |
| :--- | ---: | ---: | ---: |
| poisson | 0.0600 | 0.1420 | 125 |
| ridge | 0.0601 | 0.1410 | 125 |
| linear_regression | 0.0603 | 0.1412 | 125 |
| hist_gradient_boosting | 0.0728 | 0.1495 | 125 |
| random_forest | 0.0903 | 0.1479 | 125 |
| naive_lag1 | 0.0983 | 0.0731 | 125 |

## Melhor modelo

- **Modelo vencedor:** `poisson`
- **Arquivo de previsões por hexágono:** `C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\data\processed\cvli_hex_forecasts.csv`

Cada previsão é individualizada por hexágono, usando a série semanal histórica daquela célula.