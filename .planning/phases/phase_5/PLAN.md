# Plano de Fase: Fase 5 - Avaliação Comparativa Final e Relatório

## Objetivo
Consolidar a comparação de erros preditivos das três divisões espaciais estudadas: Bairros, Áreas Integradas de Segurança (AIS) e o Mosaico Hexagonal Ótimo gerado pelo Algoritmo Genético. Produzir o relatório estatístico final, gerando a conclusão quantitativa do trabalho solicitada pelo orientador.

---

## Modificações Propostas

### 1. Script de Avaliação Comparativa Final
#### [NEW] [src/evaluate_comparison.py](file:///C:/Users/Boanerges/Desktop/Projetos/Crime_Predict-Algoritmo%20Genetico/src/evaluate_comparison.py)
Este script executará o pipeline final:
- Carregar o dataset limpo e os parâmetros ótimos salvos em `data/processed/best_hex_grid.json`.
- Mapear os crimes para o grid hexagonal ótimo.
- Executar a predição semanal por RLM para AIS, Bairros e Mosaico Ótimo.
- Calcular os erros absolutos médios e o EQM total de previsão para cada divisão espacial no conjunto de teste.
- Imprimir o relatório comparativo estruturado e salvar os resultados em Markdown no diretório raiz do projeto.

---

## Plano de Verificação

### Teste Automatizado
- Executar o script `evaluate_comparison.py`.
- Garantir que todos os três valores de EQM são impressos na tela de forma correta e sem erros de execução.

---

## Passos para Execução
1. Criar `src/evaluate_comparison.py`.
2. Executar o script.
3. Gerar o arquivo final do Relatório Comparativo.
4. Atualizar o `STATE.md`.
