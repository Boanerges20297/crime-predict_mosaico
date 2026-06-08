import json
import os

from experiment_protocol import (
    build_experiment_summary,
    build_weekly_series_for_regions,
    choose_best_model,
    evaluate_univariate_models_per_region,
    save_protocol_outputs,
    select_top_regions_by_coverage,
)
from grid_generator import HexagonalGrid
from spatial_utils import (
    build_study_area,
    compute_bbox,
    filter_cvli,
    filter_fortaleza_bbox,
    get_processed_data_path,
    load_processed_crimes,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEST_GRID_PATH = os.path.join(BASE_DIR, "data", "processed", "best_hex_grid.json")
OUTPUT_PREFIX = "experimento_hexagonos_95"
DEFAULT_MODELS = ["naive_lag1", "linear_regression", "ridge", "poisson"]
DEFAULT_LAGS = 3
DEFAULT_TRAIN_SPLIT_DATE = "2025-12-31"


def load_best_grid():
    with open(BEST_GRID_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    data_path = get_processed_data_path()
    print(f"Carregando base para experimento hexagonal: {data_path}")
    df = load_processed_crimes()
    df = filter_cvli(df)
    df["data"] = df["data"]
    df = filter_fortaleza_bbox(df)

    best_hex = load_best_grid()
    bbox = compute_bbox(df)
    study_area = build_study_area(df)
    grid = HexagonalGrid(
        bbox,
        dx=best_hex["dx"],
        dy=best_hex["dy"],
        theta=best_hex["theta"],
        R=best_hex["R"],
        study_area=study_area,
    )

    df_hex = df.copy()
    df_hex["hex_id"] = grid.assign_points(df_hex)
    df_hex = df_hex[df_hex["hex_id"] != -1].copy()

    selected_hexes, coverage_df = select_top_regions_by_coverage(df_hex, region_col="hex_id", coverage=0.95)
    weekly_series = build_weekly_series_for_regions(df_hex, region_col="hex_id", selected_regions=selected_hexes, lags=DEFAULT_LAGS)

    experiment_results = evaluate_univariate_models_per_region(
        weekly_series,
        region_col="hex_id",
        selected_models=DEFAULT_MODELS,
        lags=DEFAULT_LAGS,
        train_split_date=DEFAULT_TRAIN_SPLIT_DATE,
    )
    best_model_name = choose_best_model(experiment_results, metric="mse_mean")
    if best_model_name is None:
        raise RuntimeError("Nenhum modelo conseguiu gerar métricas válidas para o experimento por hexágonos.")

    best_payload = experiment_results[best_model_name]
    summary_payload = build_experiment_summary(
        experiment_name="experimento_hexagonos_95",
        region_col="hex_id",
        selected_regions=selected_hexes,
        coverage_table=coverage_df,
        best_model_name=best_model_name,
        best_summary=best_payload["summary"],
        total_events=len(df_hex),
        filtered_events=int(df_hex[df_hex["hex_id"].isin(selected_hexes)].shape[0]),
        lags=DEFAULT_LAGS,
        train_split_date=DEFAULT_TRAIN_SPLIT_DATE,
        data_path=data_path,
        extra={
            "hex_grid": {
                "dx": best_hex["dx"],
                "dy": best_hex["dy"],
                "theta": best_hex["theta"],
                "R": best_hex["R"],
                "hex_count": best_hex.get("hex_count"),
                "active_hex_count": best_hex.get("active_hex_count"),
                "target_hex_count": best_hex.get("target_hex_count"),
            },
            "evaluated_models": DEFAULT_MODELS,
        },
    )

    outputs = save_protocol_outputs(
        output_prefix=OUTPUT_PREFIX,
        weekly_series_df=weekly_series,
        coverage_df=coverage_df,
        best_region_metrics_df=best_payload["region_metrics"],
        best_predictions_df=best_payload["predictions"],
        summary_payload=summary_payload,
        report_title="Experimento por Hexágonos com Cobertura de 95%",
        region_label="hexágonos",
    )

    print("\n=== Experimento por hexágonos concluído ===")
    print(f"Hexágonos selecionados: {summary_payload['selected_regions']}")
    print(f"Cobertura atingida: {summary_payload['achieved_coverage']:.4f}")
    print(f"Melhor modelo: {best_model_name}")
    print(f"MSE médio: {summary_payload['mse_mean']}")
    print(f"Relatório salvo em: {outputs['report_path']}")


if __name__ == "__main__":
    main()
