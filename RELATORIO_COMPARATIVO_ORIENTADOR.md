# Relatório Comparativo do Protocolo do Orientador

| Cenário | Unidades retidas | Cobertura | Melhor modelo | Média MSE | Variância MSE | Média MAE | Variância MAE |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| bairros_95 | 86 | 0.9523 | poisson | 0.079028 | 0.013043 | 0.184921 | 0.019942 |
| hexagonos_95 | 94 | 0.9520 | poisson | 0.075237 | 0.010838 | 0.176929 | 0.012739 |

## Conclusão global

- **Melhor cenário pelo MSE médio:** `hexagonos_95`
- **Modelo do cenário vencedor:** `poisson`

A comparação acima segue o mesmo protocolo para bairros e hexágonos: retenção das unidades que cobrem 95% dos eventos e previsão univariada por unidade espacial.