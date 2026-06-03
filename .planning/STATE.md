# Estado do Projeto

## Resumo Atual
- **Projeto inicializado:** Git configurado.
- **Base de dados de referência localizada e tratada:** `data/processed/fortaleza_crimes.csv` criada com sucesso e sem nulos em AIS.
- **Pipeline de Série Temporal e RLM implementado:** Predições de crimes semanais validadas cronologicamente. Erros de base (baseline) calculados:
  - **AIS (18 regiões ativas):** EQM = 340.8692
  - **Bairros (173 regiões ativas):** EQM = 15.2377
- **Fase atual:** Fase 4 (Algoritmo Genético de Busca do Grid Ótimo).

## Progresso das Fases
- [x] Fase 1: Engenharia de Dados & Alinhamento Territorial
- [x] Fase 2: Definição da Área de Interesse (AoI) e Grade Hexagonal Dinâmica
- [x] Fase 3: Pipeline de Previsão Semanal por Regressão Linear Múltipla
- [ ] Fase 4: Algoritmo Genético de Busca do Grid Ótimo
- [ ] Fase 5: Avaliação Comparativa Final e Relatório
