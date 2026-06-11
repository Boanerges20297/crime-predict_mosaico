# Relatório Comparativo do Protocolo do Orientador

| Cenário | Unidades retidas | Cobertura | Melhor modelo | Média MSE | Variância MSE | Média MAE | Variância MAE |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| bairros_95 | 86 | 0.9522 | poisson | 0.081781 | 0.015371 | 0.187594 | 0.020896 |
| hexagonos_95 | 94 | 0.9519 | poisson | 0.077519 | 0.013910 | 0.178733 | 0.013848 |

## Conclusão global

- **Melhor cenário pelo MSE médio:** `hexagonos_95`
- **Modelo do cenário vencedor:** `poisson`

A comparação acima segue o mesmo protocolo para bairros e hexágonos: retenção das unidades que cobrem 95% dos eventos e previsão univariada por unidade espacial.