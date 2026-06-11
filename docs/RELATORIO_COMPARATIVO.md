# Relatório Final: Otimização de Mosaico Hexagonal para Previsão de Crimes

Estudo comparativo de previsão espaço-temporal do número de crimes semanais na Cidade de Fortaleza/CE utilizando Regressão Linear Múltipla (RLM) baseada em lags temporais de 3 semanas.

## Base Utilizada
- **Arquivo:** C:\Users\Boanerges\Desktop\Projetos\Crime_Predict-Algoritmo Genetico\data\processed\fortaleza_crimes_normalizado.csv

## Parâmetros Ótimos do Grid Hexagonal (Encontrados via AG)
- **Deslocamento X (dx):** 0.832443
- **Deslocamento Y (dy):** 0.167271
- **Rotação (theta):** 0.190407 rad (10.91 graus)
- **Raio (R):** 0.006411 graus (aproximadamente 713.5 metros)

## Tabela Comparativa de Erros (EQM)

| Divisão Territorial | Regiões Ativas | Erro Quadrático Médio (EQM) |
| :--- | :---: | :---: |
| **Áreas Integradas de Segurança (AIS)** | 10 | 605.4735 |
| **Bairros** | 130 | 20.1127 |
| **Mosaico Hexagonal Ótimo** | 295 | 7.1894 |

## Conclusões
O **Mosaico Hexagonal Ótimo** obtido através da busca evolucionária obteve o menor Erro Quadrático Médio (EQM) de previsão (**7.1894**), superando tanto a divisão tradicional por Bairros (**20.1127**) quanto a divisão por Áreas Integradas de Segurança (AIS) (**605.4735**).
