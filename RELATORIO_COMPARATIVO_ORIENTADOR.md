# Relatório Comparativo do Protocolo do Orientador

| Cenário | Unidades retidas | Cobertura | Melhor modelo | Média MSE | Variância MSE | Média MAE | Variância MAE |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| bairros_95 | 86 | 0.9523 | poisson | 0.079028 | 0.013043 | 0.184921 | 0.019942 |
| hexagonos_95 | 129 | 0.9520 | ridge | 0.047603 | 0.004601 | 0.131709 | 0.007500 |

## Conclusão global

- **Melhor cenário pelo MSE médio:** `hexagonos_95`
- **Modelo do cenário vencedor:** `ridge`

A comparação acima segue o mesmo protocolo para bairros e hexágonos: retenção das unidades que cobrem 95% dos eventos e previsão univariada por unidade espacial.