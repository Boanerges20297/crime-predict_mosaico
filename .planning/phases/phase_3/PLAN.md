# Plano de Fase: Fase 3 - Pipeline de Previsão Semanal por Regressão Linear Múltipla (RLM)

## Objetivo
Desenvolver o pipeline que agrupa os dados de crime semanalmente e por partição espacial (Bairro, AIS ou Célula Hexagonal) e treina/avalia um modelo de Regressão Linear Múltipla (RLM) baseado em lags temporais (por exemplo, crimes nas semanas $t-1, t-2, t-3$ para prever crimes na semana $t$). Implementar validação cruzada temporal (walk-forward split) para medir o Erro Quadrático Médio (EQM) de forma justa e sem vazamento de dados.

---

## Modificações Propostas

### 1. Script do pipeline preditivo
#### [NEW] [src/predictor.py](file:///C:/Users/Boanerges/Desktop/Projetos/Crime_Predict-Algoritmo%20Genetico/src/predictor.py)
Este arquivo terá as seguintes responsabilidades:
- **Agregador Temporal**: Receber a base de crimes de Fortaleza e agrupar o volume de crimes semanalmente por ID de célula (Bairro, AIS ou Hexágono).
- **Geração de Lags**: Gerar os recursos de defasagem (lags) temporais (ex: 3 semanas anteriores).
- **Modelo RLM**: Treinar e testar um modelo de Regressão Linear Múltipla (`sklearn.linear_model.LinearRegression`).
- **Validação Walk-Forward**: Fazer um split cronológico (ex: usar dados até o final de 2025 para treino e o ano de 2026 para teste) para avaliar a performance por meio do Erro Quadrático Médio (EQM) médio das células ativas.

---

## Plano de Verificação

### Teste Automatizado
- Executar o predictor sobre partições estáticas (como Bairro ou AIS).
- Verificar se o script reporta corretamente o EQM global.
- Garantir que não há vazamento temporal e que as dimensões de treino e teste estão corretas.

---

## Passos para Execução
1. Criar `src/predictor.py`.
2. Escrever script de teste e verificar a integração.
3. Atualizar o `STATE.md`.
