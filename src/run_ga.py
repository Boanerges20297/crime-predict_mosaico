import json

from genetic_algorithm import GeneticAlgorithmHex
from spatial_utils import build_study_area, compute_bbox, filter_cvli, filter_fortaleza_bbox, get_processed_data_path, load_processed_crimes


def main():
    data_path = get_processed_data_path()
    print(f"Carregando crimes de {data_path}...")
    df = load_processed_crimes()
    df = filter_cvli(df)
    df_clean = filter_fortaleza_bbox(df)
    print(f"Registros CVLI válidos em Fortaleza: {len(df_clean)}")

    bbox = compute_bbox(df_clean)
    study_area = build_study_area(df_clean)

    print("\nExecutando Algoritmo Genético Piloto...")
    ga = GeneticAlgorithmHex(
        df=df_clean,
        bbox=bbox,
        study_area=study_area,
        pop_size=6,
        generations=5,
        mutation_rate=0.15,
        crossover_rate=0.8,
        seed=42,
        target_hex_count=180,
        hex_penalty_weight=5.5,
    )

    best_config = ga.run()

    print("\n[Melhor Configuração Encontrada pelo AG]")
    print(f"Deslocamento X (dx): {best_config['dx']:.6f}")
    print(f"Deslocamento Y (dy): {best_config['dy']:.6f}")
    print(f"Rotação (theta - rad): {best_config['theta']:.6f} ({best_config['theta'] * 180 / 3.14159:.2f} graus)")
    print(f"Tamanho do Raio (R - graus): {best_config['R']:.6f}")
    print(f"Menor Erro Quadrático Médio (EQM/MSE): {best_config['best_mse']:.6f}")
    print(f"Hexágonos criados: {best_config['hex_count']}")
    print(f"Hexágonos ativos: {best_config['active_hex_count']}")
    print(f"Meta de hexágonos: {best_config['target_hex_count']}")
    print(f"EQM penalizado: {best_config['penalized_mse']:.6f}")

    with open("data/processed/best_hex_grid.json", "w", encoding="utf-8") as file:
        json.dump(best_config, file, indent=4)
    print("Configuração ótima salva em data/processed/best_hex_grid.json")


if __name__ == "__main__":
    main()
