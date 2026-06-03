import pandas as pd
import numpy as np
import os

ENRICHED_PATH = r"C:\Users\Boanerges\Desktop\Projetos\Report Preview\data\raw\dados_status_ocorrencias_gerais_ENRIQUECIDO.csv"
OUTPUT_PATH = r"data/processed/fortaleza_crimes.csv"

def main():
    print("Iniciando processamento dos dados...")
    
    # 1. Carregar a base enriquecida
    if not os.path.exists(ENRICHED_PATH):
        raise FileNotFoundError(f"Arquivo enriquecido não encontrado em: {ENRICHED_PATH}")
        
    print(f"Lendo dados de {ENRICHED_PATH}...")
    df = pd.read_csv(ENRICHED_PATH, sep=',', low_memory=False)
    
    # 2. Filtrar Fortaleza
    print("Filtrando registros de Fortaleza/CE...")
    # Restringir para cidade Fortaleza e garantir que a AIS (quando informada) seja uma das 10 da capital
    # Isso remove distritos da Região Metropolitana de Fortaleza (RMF) incorretamente classificados
    capital_ais = ['AIS 05', 'AIS 06', 'AIS 08', 'AIS 16', 'AIS 17', 'AIS 18', 'AIS 19', 'AIS 20', 'AIS 21', 'AIS 22']
    df_fort = df[
        (df['cidade'].str.upper() == 'FORTALEZA') & 
        (df['ais'].isin(capital_ais) | df['ais'].isnull())
    ].copy()
    
    # Remover explicitamente bairros conhecidos da RMF que possuam erros geográficos/cadastrais na base
    rmf_bairros = ['MARECHAL RONDON', 'INDUSTRIAL', 'PARQUE LEBLON', 'IPARANA', 'PACAJUS', 'ITAITINGA', 
                   'PACATUBA', 'HORIZONTE', 'PINDORETAMA', 'PARQUE SOLEDADE', 'PARQUE ALBANO', 'TABAPU', 
                   'PARQUE DAS NACOES', 'CARARU', 'URUCUTUBA', 'DIF III', 'SENADOR POMPEU', 
                   'JUAZEIRO DO NORTE', 'PIRES FERREIRA', 'IBICUITINGA', 'REDENCAO']
    df_fort = df_fort[~df_fort['bairro'].str.upper().isin(rmf_bairros)].copy()
    
    print(f"Encontrados {len(df_fort)} registros legítimos de Fortaleza.")
    
    # 3. Converter tipos de dados básicos
    df_fort['latitude'] = pd.to_numeric(df_fort['latitude'], errors='coerce')
    df_fort['longitude'] = pd.to_numeric(df_fort['longitude'], errors='coerce')
    df_fort['data'] = pd.to_datetime(df_fort['data'], errors='coerce')
    
    # Remover registros sem coordenadas básicas válidas (se existirem)
    initial_len = len(df_fort)
    df_fort = df_fort.dropna(subset=['latitude', 'longitude'])
    removed_coords = initial_len - len(df_fort)
    if removed_coords > 0:
        print(f"Removidos {removed_coords} registros com coordenadas inválidas.")
        
    # 4. Resolver as AIS nulas
    print("Processando associação Bairro -> AIS...")
    
    # Criar mapeamento bairro -> ais a partir dos dados não nulos
    non_null_ais = df_fort[df_fort['ais'].notnull()]
    bairro_ais_map = non_null_ais.groupby('bairro')['ais'].agg(lambda x: x.value_counts().index[0]).to_dict()
    
    # Adicionar correções manuais ou conhecidas se necessário
    # Gentilândia está no bairro Benfica (AIS 05)
    bairro_ais_map['GENTILANDIA'] = 'AIS 05'
    # Sapiranga Coité está na AIS 19 ou AIS 21 (calcularemos via proximidade)
    
    # Para bairros que ainda estão sem mapeamento de AIS, vamos obter os centróides dos bairros mapeados
    # para realizar um vizinho mais próximo baseado em coordenadas.
    bairros_com_ais = df_fort[df_fort['bairro'].isin(bairro_ais_map.keys())]
    centroids = bairros_com_ais.groupby('bairro')[['latitude', 'longitude']].mean()
    centroids['ais'] = centroids.index.map(bairro_ais_map)
    
    def get_closest_ais(row):
        # Retorna a AIS do centróide de bairro catalogado mais próximo
        lat, lon = row['latitude'], row['longitude']
        dists = np.sqrt((centroids['latitude'] - lat)**2 + (centroids['longitude'] - lon)**2)
        closest_bairro = dists.idxmin()
        return centroids.loc[closest_bairro, 'ais']
        
    # Preencher AIS nulas usando o mapeamento de dicionário primeiro
    null_mask = df_fort['ais'].isnull()
    print(f"Total de registros com AIS nula inicialmente: {null_mask.sum()}")
    
    # Aplica mapeamento pelo dicionário de bairro
    df_fort.loc[null_mask, 'ais'] = df_fort.loc[null_mask, 'bairro'].map(bairro_ais_map)
    
    # Se ainda sobrarem registros nulos (bairros não catalogados), aplica o cálculo espacial de vizinho mais próximo
    still_null_mask = df_fort['ais'].isnull()
    still_null_count = still_null_mask.sum()
    if still_null_count > 0:
        print(f"Aplicando vizinho mais próximo para {still_null_count} registros de bairros novos...")
        closest_ais_values = df_fort[still_null_mask].apply(get_closest_ais, axis=1)
        df_fort.loc[still_null_mask, 'ais'] = closest_ais_values
        
    print(f"Total de registros com AIS nula após tratamento: {df_fort['ais'].isnull().sum()}")
    
    # 5. Salvar base processada
    print(f"Salvando dados processados em {OUTPUT_PATH}...")
    df_fort.to_csv(OUTPUT_PATH, index=False)
    print("Processamento concluído com sucesso!")

if __name__ == "__main__":
    main()
