# Testes de Precisao do ST-GCN

- Modelo neural: `stgcn_poisson_nll`
- Janela temporal: `12` semanas
- Ensemble: `2` sementes
- Dimensao de entrada: `6` canais

| Motor | MSE | MAE |
| :--- | ---: | ---: |
| poisson | 0.058073 | 0.140332 |
| stgcn_poisson_nll | 0.048862 | 0.109571 |

- Ganho absoluto de MSE: `0.009210`
- Ganho percentual de MSE: `15.86%`
- Ganho absoluto de MAE: `0.030761`
- Ganho percentual de MAE: `21.92%`