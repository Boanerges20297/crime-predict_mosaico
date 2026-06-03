import pandas as pd
import json
from genetic_algorithm import GeneticAlgorithmHex

def main():
    print("Carregando crimes...")
    df = pd.read_csv("data/processed/fortaleza_crimes.csv", low_memory=False)
    
    # Filtrar Bounding Box de Fortaleza limpa
    df_clean = df[(df['latitude'] >= -3.9) & (df['latitude'] <= -3.6) & 
                  (df['longitude'] >= -38.7) & (df['longitude'] <= -38.4)].copy()
                  
    min_lon = df_clean['longitude'].min()
    max_lon = df_clean['longitude'].max()
    min_lat = df_clean['latitude'].min()
    max_lat = df_clean['latitude'].max()
    bbox = (min_lon, min_lat, max_lon, max_lat)
    
    # Rodar AG com população pequena para verificação e estabilidade temporal rápida
    print("\nExecutando Algoritmo Genético Piloto...")
    ga = GeneticAlgorithmHex(
        df=df_clean,
        bbox=bbox,
        pop_size=6,
        generations=5,
        mutation_rate=0.15,
        crossover_rate=0.8,
        seed=42
    )
    
    best_config = ga.run()
    
    print("\n[Melhor Configuração Encontrada pelo AG]")
    print(f"Deslocamento X (dx): {best_config['dx']:.6f}")
    print(f"Deslocamento Y (dy): {best_config['dy']:.6f}")
    print(f"Rotação (theta - rad): {best_config['theta']:.6f} ({best_config['theta'] * 180 / 3.14159:.2f} graus)")
    print(f"Tamanho do Raio (R - graus): {best_config['R']:.6f}")
    print(f"Menor Erro Quadrático Médio (EQM/MSE): {best_config['best_mse']:.6f}")
    
    # Salvar a melhor configuração
    with open("data/processed/best_hex_grid.json", "w") as f:
        json.dump(best_config, f, indent=4)
    print("Configuração ótima salva em data/processed/best_hex_grid.json")

if __name__ == "__main__":
    main()
