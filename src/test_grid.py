import pandas as pd
import time
from grid_generator import HexagonalGrid

def main():
    print("Carregando crimes...")
    df = pd.read_csv("data/processed/fortaleza_crimes.csv", low_memory=False)
    
    # Filtrar outliers espaciais gritantes para a definição da Bounding Box de Fortaleza
    df_clean = df[(df['latitude'] >= -3.9) & (df['latitude'] <= -3.6) & 
                  (df['longitude'] >= -38.7) & (df['longitude'] <= -38.4)]
    
    min_lon = df_clean['longitude'].min()
    max_lon = df_clean['longitude'].max()
    min_lat = df_clean['latitude'].min()
    max_lat = df_clean['latitude'].max()
    bbox = (min_lon, min_lat, max_lon, max_lat)
    print(f"Bounding Box: {bbox}")
    
    print("\nTestando grade Hexagonal Base (sem rotação/deslocamento)...")
    start = time.time()
    grid = HexagonalGrid(bbox, dx=0.0, dy=0.0, theta=0.0, R=0.01)
    print(f"Hexágonos criados: {len(grid.hexagons)}")
    
    assignments = grid.assign_points(df_clean)
    elapsed = time.time() - start
    
    assigned_count = (assignments != -1).sum()
    unassigned_count = (assignments == -1).sum()
    unique_hex_active = len(set(assignments) - {-1})
    
    print(f"Tempo de geração e mapeamento: {elapsed:.2f}s")
    print(f"Crimes atribuídos a hexágonos: {assigned_count} / {len(df_clean)}")
    print(f"Crimes não atribuídos (outliers fora da borda): {unassigned_count}")
    print(f"Quantidade de hexágonos ativos (com pelo menos 1 crime): {unique_hex_active}")
    
    print("\nTestando grade com rotação (theta = 0.5 rad) e translação (dx=0.5, dy=0.5)...")
    start_rot = time.time()
    grid_rot = HexagonalGrid(bbox, dx=0.5, dy=0.5, theta=0.5, R=0.01)
    assignments_rot = grid_rot.assign_points(df_clean)
    elapsed_rot = time.time() - start_rot
    
    assigned_count_rot = (assignments_rot != -1).sum()
    print(f"Tempo: {elapsed_rot:.2f}s")
    print(f"Crimes atribuídos (com rotação/translação): {assigned_count_rot} / {len(df_clean)}")
    
    # Validar se os mapeamentos realmente diferem devido aos novos parâmetros do grid
    diff_count = (assignments != assignments_rot).sum()
    print(f"Diferença de mapeamento entre o grid base e rotacionado: {diff_count} pontos mudaram de célula.")

if __name__ == "__main__":
    main()
