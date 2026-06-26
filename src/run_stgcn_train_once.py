import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dashboard_service import (
    DEFAULT_MIN_HEX_COUNT_RATIO,
    DEFAULT_TARGET_HEX_COUNT,
    enrich_best_config,
    load_base_dataframe,
    load_saved_best_config,
)
from grid_generator import HexagonalGrid
from predictor import prepare_weekly_series
from spatial_utils import build_study_area, compute_bbox
from stgcn_forecaster import run_stgcn_pipeline, save_stgcn_dashboard_artifacts


def main():
    df = load_base_dataframe()
    bbox = compute_bbox(df)
    study_area = build_study_area(df)
    best_config = enrich_best_config(
        load_saved_best_config(),
        bbox=bbox,
        study_area=study_area,
        target_hex_count=DEFAULT_TARGET_HEX_COUNT,
        min_hex_count_ratio=DEFAULT_MIN_HEX_COUNT_RATIO,
    )
    if best_config is None:
        raise RuntimeError("best_hex_grid.json não encontrado para treino único do ST-GCN.")

    grid = HexagonalGrid(
        bbox,
        dx=best_config["dx"],
        dy=best_config["dy"],
        theta=best_config["theta"],
        R=best_config["R"],
        study_area=study_area,
    )

    df_hex = df.copy()
    df_hex["hex_id"] = grid.assign_points(df_hex)
    df_hex = df_hex[df_hex["hex_id"] != -1].copy()
    weekly_series = prepare_weekly_series(df_hex, region_col="hex_id", lags=3)

    result = run_stgcn_pipeline(weekly_series, region_col="hex_id", grid=grid)
    result["training_scope"] = "fortaleza_base_fixa"
    result["target_hex_count"] = best_config.get("target_hex_count")
    result["min_hex_count_ratio"] = best_config.get("min_hex_count_ratio")
    result["hex_count"] = int(best_config.get("hex_count", len(grid.display_hexagons)))
    result["active_hex_count"] = int(df_hex["hex_id"].nunique())
    save_stgcn_dashboard_artifacts(result)

    print("=== Treino unico do ST-GCN concluido ===")
    print(
        {
            "model_name": result["model_name"],
            "mse": result["mse"],
            "mae": result["mae"],
            "hex_count": result["hex_count"],
            "active_hex_count": result["active_hex_count"],
            "training_scope": result["training_scope"],
        }
    )


if __name__ == "__main__":
    main()
