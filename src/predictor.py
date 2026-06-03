import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def prepare_weekly_series(df, region_col, lags=3):
    """
    Agrupa os crimes semanalmente e gera as variáveis de lag.
    """
    df = df.copy()
    df['data'] = pd.to_datetime(df['data'])
    
    # Adicionar ano-semana ou data inicial da semana correspondente
    # Usamos o período semanal iniciando na segunda-feira ou baseado no dia do ano
    df['semana'] = df['data'].dt.to_period('W').dt.start_time
    
    # Agrupar crimes por semana e região
    weekly_counts = df.groupby(['semana', region_col]).size().reset_index(name='crimes')
    
    # Criar um grid completo de semanas e regiões para evitar buracos temporais (semanas sem crime = 0)
    semanas_unicas = pd.date_range(start=weekly_counts['semana'].min(), 
                                    end=weekly_counts['semana'].max(), 
                                    freq='W-MON')
    regioes_unicas = weekly_counts[region_col].unique()
    
    multi_idx = pd.MultiIndex.from_product([semanas_unicas, regioes_unicas], names=['semana', region_col])
    grid_df = pd.DataFrame(index=multi_idx).reset_index()
    
    # Fazer merge para trazer a contagem de crimes
    merged = pd.merge(grid_df, weekly_counts, on=['semana', region_col], how='left')
    merged['crimes'] = merged['crimes'].fillna(0).astype(int)
    
    # Ordenar por região e tempo para aplicar lags
    merged = merged.sort_values(by=[region_col, 'semana']).reset_index(drop=True)
    
    # Gerar os Lags temporais
    for l in range(1, lags + 1):
        merged[f'lag_{l}'] = merged.groupby(region_col)['crimes'].shift(l)
        
    # Dropar linhas iniciais com nulos devido aos lags
    merged = merged.dropna().reset_index(drop=True)
    return merged

def evaluate_model(df_model, region_col, lags=3, train_split_date="2025-12-31"):
    """
    Treina o modelo de RLM e calcula o EQM (MSE) na validação cronológica (walk-forward).
    Utiliza dados até train_split_date para treino e depois para teste.
    """
    train_split_date = pd.to_datetime(train_split_date)
    
    train_mask = df_model['semana'] <= train_split_date
    test_mask = df_model['semana'] > train_split_date
    
    df_train = df_model[train_mask]
    df_test = df_model[test_mask]
    
    if len(df_train) == 0 or len(df_test) == 0:
        # Se os dados temporais forem insuficientes para treinar/testar
        return np.nan
        
    features = [f'lag_{l}' for l in range(1, lags + 1)]
    
    X_train = df_train[features]
    y_train = df_train['crimes']
    X_test = df_test[features]
    y_test = df_test['crimes']
    
    # Treinar a Regressão Linear Múltipla
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Fazer as predições no conjunto de teste
    predictions = model.predict(X_test)
    
    # Garantir que previsões negativas sejam ajustadas para zero
    predictions = np.clip(predictions, 0, None)
    
    # Calcular o Erro Quadrático Médio (EQM)
    mse = mean_squared_error(y_test, predictions)
    return mse
