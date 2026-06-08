import json
import os

from skforecast.recursive import ForecasterRecursive

from grid_generator import HexagonalGrid
from skforecast_protocol import run_skforecast_phase
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
OUTPUT_PREFIX = "fase2_skforecast_hexagonos_95"
SELECTED_MODELS = ["linear_regression", "ridge", "poisson", "hist_gradient_boosting"]
LAGS = 3
TRAIN_SPLIT_DATE = "2025-12-31"


def load_best_grid():
    with open(BEST_GRID_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    data_path = get_processed_data_path()
    print(f"Carregando base para fase 2 skforecast em hexagonos: {data_path}")
    df = load_processed_crimes()
    df = filter_cvli(df)
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

    result = run_skforecast_phase(
        df=df_hex,
        region_col="hex_id",
        output_prefix=OUTPUT_PREFIX,
        report_title="Fase 2 com skforecast em Hexagonos (95%)",
        region_label="hexagonos",
        data_path=data_path,
        forecaster_cls=ForecasterRecursive,
        selected_models=SELECTED_MODELS,
        lags=LAGS,
        train_split_date=TRAIN_SPLIT_DATE,
        extra_summary={
            "framework": "skforecast",
            "skforecast_strategy": "ForecasterRecursive",
            "evaluated_models": SELECTED_MODELS,
            "hex_grid": best_hex,
        },
    )

    print("\n=== Fase 2 skforecast em hexagonos concluida ===")
    print(f"Melhor modelo: {result['best_model']}")
    print(f"MSE medio: {result['summary']['mse_mean']}")
    print(f"Relatorio salvo em: {result['outputs']['report_path']}")


if __name__ == "__main__":
    main()
