import pandas as pd
import json
import folium
from folium import FeatureGroup, LayerControl
from grid_generator import HexagonalGrid

def main():
    print("Carregando base de crimes para o mapa...")
    df = pd.read_csv("data/processed/fortaleza_crimes.csv", low_memory=False)
    
    # Filtrar área de Fortaleza limpa
    df_clean = df[(df['latitude'] >= -3.9) & (df['latitude'] <= -3.6) & 
                  (df['longitude'] >= -38.7) & (df['longitude'] <= -38.4)].copy()
                  
    # Carregar melhor configuração do grid hexagonal
    with open("data/processed/best_hex_grid.json", "r") as f:
        best_hex = json.load(f)
        
    min_lon, max_lon = df_clean['longitude'].min(), df_clean['longitude'].max()
    min_lat, max_lat = df_clean['latitude'].min(), df_clean['latitude'].max()
    bbox = (min_lon, min_lat, max_lon, max_lat)
    
    # Criar grade hexagonal com os parâmetros ótimos
    grid = HexagonalGrid(bbox, dx=best_hex['dx'], dy=best_hex['dy'], theta=best_hex['theta'], R=best_hex['R'])
    df_clean['hex_id'] = grid.assign_points(df_clean)
    
    # 1. Configurar mapa base no centróide de Fortaleza
    lat_center = df_clean['latitude'].mean()
    lon_center = df_clean['longitude'].mean()
    
    # Mapa estilizado em tema escuro premium
    m = folium.Map(location=[lat_center, lon_center], zoom_start=12, tiles="cartodbpositron")
    
    # 2. Criar Feature Groups para cada modelo territorial
    group_ais = FeatureGroup(name="1. Áreas Integradas de Segurança (AIS)", show=True)
    group_bairros = FeatureGroup(name="2. Divisão por Bairros", show=False)
    group_hex = FeatureGroup(name="3. Mosaico Hexagonal Ótimo (AG)", show=False)
    
    # --- MODELO 1: AIS ---
    print("Mapeando agrupamentos de AIS...")
    ais_centroids = df_clean.groupby('ais')[['latitude', 'longitude']].mean()
    ais_counts = df_clean['ais'].value_counts()
    for ais_name, row in ais_centroids.iterrows():
        cnt = ais_counts.get(ais_name, 0)
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=15 + (cnt / 1000),
            color="#FF4B4B",
            fill=True,
            fill_color="#FF4B4B",
            fill_opacity=0.6,
            popup=f"Região: {ais_name}<br>Total Ocorrências: {cnt}"
        ).add_to(group_ais)
        
    # --- MODELO 2: Bairros ---
    print("Mapeando agrupamentos de Bairros...")
    bairro_centroids = df_clean.groupby('bairro')[['latitude', 'longitude']].mean()
    bairro_counts = df_clean['bairro'].value_counts()
    for bairro_name, row in bairro_centroids.iterrows():
        cnt = bairro_counts.get(bairro_name, 0)
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6 + (cnt / 300),
            color="#3B82F6",
            fill=True,
            fill_color="#3B82F6",
            fill_opacity=0.6,
            popup=f"Bairro: {bairro_name}<br>Total Ocorrências: {cnt}"
        ).add_to(group_bairros)
        
    # --- MODELO 3: Hexagonal Ótimo ---
    print("Mapeando polígonos do Mosaico Hexagonal Ótimo...")
    hex_counts = df_clean['hex_id'].value_counts().to_dict()
    
    # Adicionar polígonos dos hexágonos ativos ao mapa
    for hex_id, count in hex_counts.items():
        if hex_id == -1:
            continue
        poly = grid.hexagons[hex_id]
        # Coordenadas do polígono em formato [lat, lon] para o Folium
        coords = [[lat, lon] for lon, lat in poly.exterior.coords]
        
        # Coloração dependente da densidade de ocorrências
        color_intensity = min(int(50 + (count / 1500) * 200), 255)
        hex_color = f"#{color_intensity:02x}10a0"  # Variações de roxo/magenta
        
        folium.Polygon(
            locations=coords,
            color="#A855F7",
            weight=1,
            fill=True,
            fill_color=hex_color,
            fill_opacity=0.5,
            popup=f"Hexágono ID: {hex_id}<br>Ocorrências no Período: {count}"
        ).add_to(group_hex)
        
    # Adicionar grupos ao mapa
    group_ais.add_to(m)
    group_bairros.add_to(m)
    group_hex.add_to(m)
    
    # Ativar o seletor interativo de camadas
    LayerControl(collapsed=False).add_to(m)
    
    # Salvar mapa interativo
    output_html = "mapa_interativo_fortaleza.html"
    m.save(output_html)
    print(f"Mapa interativo gerado com sucesso em '{output_html}'")

if __name__ == "__main__":
    main()
