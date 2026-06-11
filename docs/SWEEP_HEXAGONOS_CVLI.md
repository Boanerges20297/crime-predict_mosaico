# Sweep de Malhas Hexagonais para CVLI

## Comparativo por malha-alvo

| Alvo | Hex criados | Hex ativos | Melhor modelo rápido | MSE | MAE | Média CVLI/hex | Mediana CVLI/hex | Média atividade | Mediana atividade |
| ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 120 | 119 | 109 | poisson | 0.0656 | 0.1619 | 28.39 | 20.00 | 0.1089 | 0.0866 |
| 140 | 133 | 125 | poisson | 0.0600 | 0.1420 | 24.76 | 18.00 | 0.0955 | 0.0779 |
| 160 | 133 | 125 | poisson | 0.0600 | 0.1420 | 24.76 | 18.00 | 0.0955 | 0.0779 |
| 180 | 133 | 125 | poisson | 0.0600 | 0.1420 | 24.76 | 18.00 | 0.0955 | 0.0779 |
| 200 | 133 | 125 | poisson | 0.0600 | 0.1420 | 24.76 | 18.00 | 0.0955 | 0.0779 |

## Malha escolhida

- **Alvo solicitado:** 140 hexágonos
- **Hexágonos realmente criados:** 133
- **Hexágonos ativos:** 125
- **Melhor modelo rápido no sweep:** `poisson`
- **Critério de escolha:** menor MSE com pequeno peso para proximidade de ~180 hexágonos