# Relatório Final: Otimização de Mosaico Hexagonal para Previsão de Crimes

Estudo comparativo de previsão espaço-temporal do número de crimes semanais na Cidade de Fortaleza/CE utilizando Regressão Linear Múltipla (RLM) baseada em lags temporais de 3 semanas.

## Parâmetros Ótimos do Grid Hexagonal (Encontrados via AG)
- **Deslocamento X (dx):** 0.832443
- **Deslocamento Y (dy):** 0.182787
- **Rotação (theta):** 0.190407 rad (10.91 graus)
- **Raio (R):** 0.006351 graus (aproximadamente 706.9 metros)

## Tabela Comparativa de Erros (EQM)

| Divisão Territorial | Regiões Ativas | Erro Quadrático Médio (EQM) |
| :--- | :---: | :---: |
| **Áreas Integradas de Segurança (AIS)** | 14 | 437.1137 |
| **Bairros** | 162 | 16.2539 |
| **Mosaico Hexagonal Ótimo** | 323 | 6.7456 |

## Conclusões
O **Mosaico Hexagonal Ótimo** obtido através da busca evolucionária obteve o menor Erro Quadrático Médio (EQM) de previsão (**6.7456**), superando tanto a divisão tradicional por Bairros (**16.2539**) quanto a divisão por Áreas Integradas de Segurança (AIS) (**437.1137**).
