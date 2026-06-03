# Plano de Fase: Fase 1 - Engenharia de Dados & Alinhamento Territorial

## Objetivo
Processar a base de dados de ocorrências criminais enriquecidas para a cidade de Fortaleza/CE, garantindo que tenhamos registros válidos, limpos e com preenchimento de 100% nas colunas espaciais cruciais (`latitude`, `longitude`, `bairro`, `ais`). Os dados ausentes na coluna `ais` serão preenchidos usando mapeamento bairro $\rightarrow$ AIS e vizinhança geográfica de menor distância para bairros desconhecidos.

---

## Modificações Propostas

### 1. Novo script de processamento de dados
#### [NEW] [prepare_data.py](file:///C:/Users/Boanerges/Desktop/Projetos/Crime_Predict-Algoritmo%20Genetico/src/prepare_data.py)
Este script carregará a base de dados de referência:
`C:/Users/Boanerges/Desktop/Projetos/Report Preview/data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv`.

**Lógica de Execução:**
1. Filtrar registros onde `cidade == 'Fortaleza'`.
2. Converter coordenadas (`latitude`, `longitude`) e datas (`data`) para os formatos corretos (`float64` e `datetime`).
3. Montar a tabela de referência de mapeamento de `bairro` para `ais` a partir dos registros que já possuem `ais` preenchido.
4. Para as linhas onde `ais` é nula (`NaN`):
   - Se o bairro do registro estiver na tabela de referência, preencher com o valor correspondente.
   - Para bairros novos/não catalogados (ex: Gentilândia, Sapiranga Coité): Calcular a distância euclidiana média das coordenadas do crime até as coordenadas médias (centróides) dos bairros catalogados, associando à AIS do bairro catalogado mais próximo.
5. Salvar o arquivo resultante em `data/processed/fortaleza_crimes.csv` no diretório do projeto.

---

## Plano de Verificação

### Teste Automatizado
Podemos rodar um script rápido para validar o dataset de saída:
- Verificar se `data/processed/fortaleza_crimes.csv` existe.
- Verificar se o número de linhas com `cidade == 'Fortaleza'` é condizente (~112.318).
- Verificar se a coluna `ais` e a coluna `bairro` possuem zero valores nulos (`0`).
- Verificar se as coordenadas são do tipo `float64` e estão dentro dos limites geográficos aproximados de Fortaleza.

---

## Passos para Execução
1. Criar diretório `src/` e o script `prepare_data.py`.
2. Executar o script.
3. Verificar a saída.
4. Atualizar o `STATE.md`.
