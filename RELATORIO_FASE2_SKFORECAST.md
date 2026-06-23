# Relatorio da Fase 2 com skforecast

| Cenario | Unidades retidas | Cobertura | Melhor modelo | Media MSE | Variancia MSE | Media MAE | Variancia MAE |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| fase2_skforecast_bairros_95 | 86 | 0.9522 | poisson | 0.082041 | 0.014875 | 0.187150 | 0.021579 |
| fase2_skforecast_hexagonos_95 | 94 | 0.9519 | hist_gradient_boosting | 0.074178 | 0.013288 | 0.170417 | 0.012171 |

## Conclusao

- **Melhor cenario global na fase 2:** `fase2_skforecast_hexagonos_95`
- **Melhor modelo no cenario vencedor:** `hist_gradient_boosting`

A fase 2 usa `skforecast` com estrategia recursiva (`ForecasterRecursive`) e mantem o mesmo corte espacial de 95% para garantir comparabilidade com a fase 1.