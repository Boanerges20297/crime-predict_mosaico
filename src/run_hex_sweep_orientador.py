import json
import os

from experiment_protocol import (
    build_weekly_series_for_regions,
    choose_best_model,
    evaluate_univariate_models_per_region,
    select_top_regions_by_coverage,
)
from genetic_algorithm import GeneticAlgorithmHex
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
OUTPUT_JSON_PATH = os.path.join(BASE_DIR, "data", "processed", "hex_sweep_orientador.json")
OUTPUT_MD_PATH = os.path.join(BASE_DIR, "SWEEP_HEXAGONOS_ORIENTADOR.md")
BEST_GRID_PATH = os.path.join(BASE_DIR, "data", "processed", "best_hex_grid.json")
TARGETS = [120, 140, 160, 180, 200]
DEFAULT_MODELS = ["naive_lag1", "linear_regression", "ridge", "poisson"]
DEFAULT_LAGS = 3
DEFAULT_TRAIN_SPLIT_DATE = "2025-12-31"


def run_target(df, bbox, study_area, target_hex_count):
    ga = GeneticAlgorithmHex(
        df=df,
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
    best_payload = experiment_results[best_model_name]

    return {
        "target_hex_count": target_hex_count,
        "best_grid": best,
        "created_hex_count": int(best["hex_count"]),
        "active_hex_count": int(best["active_hex_count"]),
        "retained_hex_count": int(len(selected_hexes)),
        "coverage_achieved": float(df_hex[df_hex["hex_id"].isin(selected_hexes)].shape[0] / max(len(df_hex), 1)),
        "best_model": best_model_name,
        "mse_mean": best_payload["summary"]["mse_mean"],
        "mse_variance": best_payload["summary"]["mse_variance"],
        "mae_mean": best_payload["summary"]["mae_mean"],
        "mae_variance": best_payload["summary"]["mae_variance"],
        "coverage_preview": coverage_df.head(5).to_dict(orient="records"),
    }


def save_outputs(rows, chosen):
    payload = {
        "targets": rows,
        "chosen_target": chosen["target_hex_count"],
        "chosen_created_hex_count": chosen["created_hex_count"],
        "chosen_best_model": chosen["best_model"],
    }
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    lines = [
        "# Sweep de Hexagonos no Protocolo do Orientador",
        "",
        "| Alvo | Hex criados | Hex ativos | Hex retidos 95% | Cobertura | Melhor modelo | Media MSE | Variancia MSE | Media MAE | Variancia MAE |",
        "| ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['target_hex_count']} | {row['created_hex_count']} | {row['active_hex_count']} | "
            f"{row['retained_hex_count']} | {row['coverage_achieved']:.4f} | {row['best_model']} | "
            f"{row['mse_mean']:.6f} | {row['mse_variance']:.6f} | {row['mae_mean']:.6f} | {row['mae_variance']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Malha escolhida",
            "",
            f"- **Alvo escolhido:** {chosen['target_hex_count']}",
            f"- **Hexagonos criados:** {chosen['created_hex_count']}",
            f"- **Hexagonos ativos:** {chosen['active_hex_count']}",
            f"- **Hexagonos retidos no corte de 95%:** {chosen['retained_hex_count']}",
            f"- **Melhor modelo:** `{chosen['best_model']}`",
        ]
    )

    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main():
    data_path = get_processed_data_path()
    print(f"Carregando base para sweep do protocolo do orientador: {data_path}")
    df = load_processed_crimes()
    df = filter_cvli(df)
    df = filter_fortaleza_bbox(df)

    bbox = compute_bbox(df)
    study_area = build_study_area(df)

    rows = []
    for target in TARGETS:
        print(f"\n=== Sweep orientador para alvo {target} ===")
        result = run_target(df, bbox, study_area, target)
        print(
            f"Alvo {target} -> criados={result['created_hex_count']} | "
            f"retidos95={result['retained_hex_count']} | "
            f"mse_medio={result['mse_mean']:.6f}"
        )
        rows.append(result)

    chosen = min(rows, key=lambda item: item["mse_mean"])
    save_outputs(rows, chosen)

    with open(BEST_GRID_PATH, "w", encoding="utf-8") as file:
        json.dump(chosen["best_grid"], file, indent=2)

    print("\nSweep do orientador concluido.")
    print(f"Melhor alvo: {chosen['target_hex_count']}")
    print(f"Relatorio salvo em: {OUTPUT_MD_PATH}")


if __name__ == "__main__":
    main()
