import json

from benchmark_models import benchmark_models, save_benchmark_outputs, train_best_model_and_forecast
from grid_generator import HexagonalGrid
from predictor import prepare_weekly_series
from spatial_utils import build_study_area, compute_bbox, filter_cvli, filter_fortaleza_bbox, get_processed_data_path, load_processed_crimes


def main():
    data_path = get_processed_data_path()
    print(f"Carregando base para benchmark: {data_path}")
    df = load_processed_crimes()
    df = filter_cvli(df)
    df_clean = filter_fortaleza_bbox(df)

    with open("data/processed/best_hex_grid.json", "r", encoding="utf-8") as file:
        best_hex = json.load(file)

    bbox = compute_bbox(df_clean)
    study_area = build_study_area(df_clean)
    grid = HexagonalGrid(
        bbox,
        dx=best_hex["dx"],
        dy=best_hex["dy"],
        theta=best_hex["theta"],
        R=best_hex["R"],
        study_area=study_area,
    )

    df_clean["hex_id"] = grid.assign_points(df_clean)
    df_hex = df_clean[df_clean["hex_id"] != -1].copy()
    weekly_series = prepare_weekly_series(df_hex, region_col="hex_id", lags=3)

    print(f"Hexágonos criados: {len(grid.display_hexagons)}")
    print(f"Hexágonos ativos: {df_hex['hex_id'].nunique()}")
    print("Executando benchmark...")
    ranking = benchmark_models(weekly_series, region_col="hex_id", lags=3)

    for row in ranking:
        print(
            f"{row['model']:<24} | MSE={row['mse']:.4f} | "
            f"MAE={row['mae']:.4f} | hex={row['hexagons_evaluated']}"
        )

    best_model_name, forecasts_df = train_best_model_and_forecast(weekly_series, ranking, region_col="hex_id", lags=3)
    save_benchmark_outputs(ranking, best_model_name, forecasts_df)
    print(f"\nMelhor modelo: {best_model_name}")
    print("Resultados salvos em BENCHMARK_CVLI_HEXAGONOS.md e data/processed/cvli_hex_forecasts.csv")


if __name__ == "__main__":
    main()
