import json
import os

import pandas as pd

from benchmark_models import benchmark_models, save_benchmark_outputs, train_best_model_and_forecast
from genetic_algorithm import GeneticAlgorithmHex
from grid_generator import HexagonalGrid
from predictor import prepare_weekly_series
from spatial_utils import build_study_area, compute_bbox, filter_cvli, filter_fortaleza_bbox, get_processed_data_path, load_processed_crimes


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP_JSON_PATH = os.path.join(BASE_DIR, "data", "processed", "cvli_hex_sweep.json")
SWEEP_MD_PATH = os.path.join(BASE_DIR, "SWEEP_HEXAGONOS_CVLI.md")
WEEKLY_SERIES_CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "cvli_hex_weekly_series.csv")
TARGETS = [120, 140, 160, 180, 200]
QUICK_MODELS = ["naive_lag1", "linear_regression", "ridge", "poisson"]


def compute_hex_activity_stats(df_hex):
    temp = df_hex.copy()
    temp["semana"] = pd.to_datetime(temp["data"]).dt.to_period("W").dt.start_time
    weekly = temp.groupby(["hex_id", "semana"]).size().reset_index(name="cvli")
    total_weeks = int(weekly["semana"].nunique()) if not weekly.empty else 0
    per_hex = weekly.groupby("hex_id").agg(total_cvli=("cvli", "sum"), active_weeks=("semana", "nunique")).reset_index()
    if per_hex.empty:
        return {
            "mean_total_cvli_per_hex": 0.0,
            "median_total_cvli_per_hex": 0.0,
            "mean_activity_ratio": 0.0,
            "median_activity_ratio": 0.0,
        }

    per_hex["activity_ratio"] = per_hex["active_weeks"] / max(total_weeks, 1)
    return {
        "mean_total_cvli_per_hex": float(per_hex["total_cvli"].mean()),
        "median_total_cvli_per_hex": float(per_hex["total_cvli"].median()),
        "mean_activity_ratio": float(per_hex["activity_ratio"].mean()),
        "median_activity_ratio": float(per_hex["activity_ratio"].median()),
    }


def run_target(df_clean, bbox, study_area, target_hex_count):
    ga = GeneticAlgorithmHex(
        df=df_clean,
        bbox=bbox,
        study_area=study_area,
        pop_size=6,
        generations=5,
        mutation_rate=0.15,
        crossover_rate=0.8,
        seed=42,
        target_hex_count=target_hex_count,
        hex_penalty_weight=5.5,
    )
    best = ga.run()

    grid = HexagonalGrid(
        bbox,
        dx=best["dx"],
        dy=best["dy"],
        theta=best["theta"],
        R=best["R"],
        study_area=study_area,
    )
    df_hex = df_clean.copy()
    df_hex["hex_id"] = grid.assign_points(df_hex)
    df_hex = df_hex[df_hex["hex_id"] != -1].copy()
    weekly_series = prepare_weekly_series(df_hex, region_col="hex_id", lags=3)

    quick_ranking = benchmark_models(weekly_series, region_col="hex_id", lags=3, selected_models=QUICK_MODELS)
    best_quick = quick_ranking[0] if quick_ranking else None
    stability = compute_hex_activity_stats(df_hex)

    return {
        "target_hex_count": target_hex_count,
        "best_config": best,
        "grid": grid,
        "df_hex": df_hex,
        "weekly_series": weekly_series,
        "quick_ranking": quick_ranking,
        "best_quick_model": best_quick["model"] if best_quick else None,
        "best_quick_mse": best_quick["mse"] if best_quick else None,
        "best_quick_mae": best_quick["mae"] if best_quick else None,
        "stability": stability,
    }


def select_best_result(results):
    scored = []
    for result in results:
        mse = result["best_quick_mse"]
        achieved_hex_count = result["best_config"]["hex_count"]
        distance_penalty = abs(achieved_hex_count - 180) / 180.0
        combined_score = mse + 0.05 * distance_penalty
        scored.append((combined_score, result))

    scored.sort(key=lambda item: item[0])
    return scored[0][1]


def save_sweep_outputs(results, chosen):
    rows = []
    for result in results:
        rows.append(
            {
                "target_hex_count": result["target_hex_count"],
                "achieved_hex_count": result["best_config"]["hex_count"],
                "active_hex_count": result["best_config"]["active_hex_count"],
                "best_quick_model": result["best_quick_model"],
                "best_quick_mse": result["best_quick_mse"],
                "best_quick_mae": result["best_quick_mae"],
                "mean_total_cvli_per_hex": result["stability"]["mean_total_cvli_per_hex"],
                "median_total_cvli_per_hex": result["stability"]["median_total_cvli_per_hex"],
                "mean_activity_ratio": result["stability"]["mean_activity_ratio"],
                "median_activity_ratio": result["stability"]["median_activity_ratio"],
                "radius": result["best_config"]["R"],
                "theta": result["best_config"]["theta"],
            }
        )

    payload = {
        "targets": rows,
        "chosen_target": chosen["target_hex_count"],
        "chosen_achieved_hex_count": chosen["best_config"]["hex_count"],
        "chosen_best_quick_model": chosen["best_quick_model"],
    }
    with open(SWEEP_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    lines = [
        "# Sweep de Malhas Hexagonais para CVLI",
        "",
        "## Comparativo por malha-alvo",
        "",
        "| Alvo | Hex criados | Hex ativos | Melhor modelo rápido | MSE | MAE | Média CVLI/hex | Mediana CVLI/hex | Média atividade | Mediana atividade |",
        "| ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['target_hex_count']} | {row['achieved_hex_count']} | {row['active_hex_count']} | "
            f"{row['best_quick_model']} | {row['best_quick_mse']:.4f} | {row['best_quick_mae']:.4f} | "
            f"{row['mean_total_cvli_per_hex']:.2f} | {row['median_total_cvli_per_hex']:.2f} | "
            f"{row['mean_activity_ratio']:.4f} | {row['median_activity_ratio']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Malha escolhida",
            "",
            f"- **Alvo solicitado:** {chosen['target_hex_count']} hexágonos",
            f"- **Hexágonos realmente criados:** {chosen['best_config']['hex_count']}",
            f"- **Hexágonos ativos:** {chosen['best_config']['active_hex_count']}",
            f"- **Melhor modelo rápido no sweep:** `{chosen['best_quick_model']}`",
            f"- **Critério de escolha:** menor MSE com pequeno peso para proximidade de ~180 hexágonos",
        ]
    )

    with open(SWEEP_MD_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main():
    data_path = get_processed_data_path()
    print(f"Carregando base para sweep: {data_path}")
    df = load_processed_crimes()
    df = filter_cvli(df)
    df_clean = filter_fortaleza_bbox(df)

    bbox = compute_bbox(df_clean)
    study_area = build_study_area(df_clean)

    results = []
    for target in TARGETS:
        print(f"\n=== Sweep para alvo {target} hexágonos ===")
        result = run_target(df_clean, bbox, study_area, target)
        print(
            f"Alvo {target} -> criados={result['best_config']['hex_count']} | "
            f"ativos={result['best_config']['active_hex_count']} | "
            f"melhor={result['best_quick_model']} | mse={result['best_quick_mse']:.4f}"
        )
        results.append(result)

    chosen = select_best_result(results)
    save_sweep_outputs(results, chosen)

    chosen_weekly = chosen["weekly_series"]
    chosen_weekly.to_csv(WEEKLY_SERIES_CSV_PATH, index=False)

    full_ranking = benchmark_models(chosen_weekly, region_col="hex_id", lags=3)
    best_model_name, forecasts_df = train_best_model_and_forecast(chosen_weekly, full_ranking, region_col="hex_id", lags=3)
    save_benchmark_outputs(full_ranking, best_model_name, forecasts_df)

    with open(os.path.join(BASE_DIR, "data", "processed", "best_hex_grid.json"), "w", encoding="utf-8") as file:
        json.dump(chosen["best_config"], file, indent=4)

    print("\n=== Resultado final ===")
    print(f"Malha escolhida: alvo={chosen['target_hex_count']} | criados={chosen['best_config']['hex_count']} | ativos={chosen['best_config']['active_hex_count']}")
    print(f"Melhor modelo final: {best_model_name}")
    print(f"Sweep salvo em: {SWEEP_MD_PATH}")
    print(f"Séries semanais salvas em: {WEEKLY_SERIES_CSV_PATH}")
    print("Previsões por hexágono salvas em: data/processed/cvli_hex_forecasts.csv")


if __name__ == "__main__":
    main()
