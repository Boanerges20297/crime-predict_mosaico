import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(BASE_DIR, "src") not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from benchmark_models import benchmark_models
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
from stgcn_forecaster import run_stgcn_pipeline, save_stgcn_precision_report


def build_hex_dataset():
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
        raise RuntimeError("best_hex_grid.json não encontrado para o teste de precisão.")

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
    return weekly_series, grid


def main():
    weekly_series, grid = build_hex_dataset()

    poisson_ranking = benchmark_models(
        weekly_series,
        region_col="hex_id",
        lags=3,
        selected_models=["poisson"],
    )
    if not poisson_ranking:
        raise RuntimeError("Não foi possível gerar baseline Poisson para os testes de precisão.")

    poisson_metrics = poisson_ranking[0]
    stgcn_result = run_stgcn_pipeline(weekly_series, region_col="hex_id", grid=grid)

    payload = {
        "poisson": {
            "mse": float(poisson_metrics["mse"]),
            "mae": float(poisson_metrics["mae"]),
        },
        "stgcn": {
            "mse": float(stgcn_result["mse"]),
            "mae": float(stgcn_result["mae"]),
            "seq_len": int(stgcn_result["seq_len"]),
            "ensemble_size": int(stgcn_result["ensemble_size"]),
            "input_dim": int(stgcn_result["input_dim"]),
            "best_validation_loss": float(stgcn_result["best_validation_loss"]),
            "node_count": int(stgcn_result["node_count"]),
        },
    }
    payload["comparison"] = {
        "mse_gain": float(payload["poisson"]["mse"] - payload["stgcn"]["mse"]),
        "mse_gain_pct": float(((payload["poisson"]["mse"] - payload["stgcn"]["mse"]) / max(payload["poisson"]["mse"], 1e-8)) * 100.0),
        "mae_gain": float(payload["poisson"]["mae"] - payload["stgcn"]["mae"]),
        "mae_gain_pct": float(((payload["poisson"]["mae"] - payload["stgcn"]["mae"]) / max(payload["poisson"]["mae"], 1e-8)) * 100.0),
    }

    save_stgcn_precision_report(payload)
    print("=== Testes de precisão do ST-GCN ===")
    print(payload)


if __name__ == "__main__":
    main()
