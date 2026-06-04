import time

from grid_generator import HexagonalGrid
from spatial_utils import build_study_area, compute_bbox, get_processed_data_path, load_processed_crimes, filter_cvli, filter_fortaleza_bbox


def main():
    data_path = get_processed_data_path()
    print(f"Carregando crimes de {data_path}...")
    df = load_processed_crimes()
    df = filter_cvli(df)
    df_clean = filter_fortaleza_bbox(df)

    bbox = compute_bbox(df_clean)
    study_area = build_study_area(df_clean)
    print(f"Bounding Box: {bbox}")

    print("\nTestando grade Hexagonal Base (sem rotação/deslocamento)...")
    start = time.time()
    grid = HexagonalGrid(bbox, dx=0.0, dy=0.0, theta=0.0, R=0.01, study_area=study_area)
    print(f"Hexágonos criados: {len(grid.hexagons)}")

    assignments = grid.assign_points(df_clean)
    elapsed = time.time() - start

    assigned_count = (assignments != -1).sum()
    unassigned_count = (assignments == -1).sum()
    unique_hex_active = len(set(assignments) - {-1})

    print(f"Tempo de geração e mapeamento: {elapsed:.2f}s")
    print(f"Crimes atribuídos a hexágonos: {assigned_count} / {len(df_clean)}")
    print(f"Crimes não atribuídos: {unassigned_count}")
    print(f"Quantidade de hexágonos ativos: {unique_hex_active}")

    print("\nTestando grade com rotação (theta = 0.5 rad) e translação (dx=0.5, dy=0.5)...")
    start_rot = time.time()
    grid_rot = HexagonalGrid(bbox, dx=0.5, dy=0.5, theta=0.5, R=0.01, study_area=study_area)
    assignments_rot = grid_rot.assign_points(df_clean)
    elapsed_rot = time.time() - start_rot

    assigned_count_rot = (assignments_rot != -1).sum()
    print(f"Tempo: {elapsed_rot:.2f}s")
    print(f"Crimes atribuídos (com rotação/translação): {assigned_count_rot} / {len(df_clean)}")

    diff_count = (assignments != assignments_rot).sum()
    print(f"Diferença de mapeamento entre o grid base e rotacionado: {diff_count} pontos mudaram de célula.")


if __name__ == "__main__":
    main()
