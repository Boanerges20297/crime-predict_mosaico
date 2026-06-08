import json
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAIRROS_SUMMARY_PATH = os.path.join(BASE_DIR, "data", "processed", "fase2_skforecast_bairros_95_summary.json")
HEX_SUMMARY_PATH = os.path.join(BASE_DIR, "data", "processed", "fase2_skforecast_hexagonos_95_summary.json")
OUTPUT_JSON_PATH = os.path.join(BASE_DIR, "data", "processed", "fase2_skforecast_comparison.json")
OUTPUT_MD_PATH = os.path.join(BASE_DIR, "RELATORIO_FASE2_SKFORECAST.md")


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    bairros = load_json(BAIRROS_SUMMARY_PATH)
    hexagonos = load_json(HEX_SUMMARY_PATH)

    rows = [
        {
            "cenario": "fase2_skforecast_bairros_95",
            "best_model": bairros["best_model"],
            "selected_regions": bairros["selected_regions"],
            "achieved_coverage": bairros["achieved_coverage"],
            "mse_mean": bairros["mse_mean"],
            "mse_variance": bairros["mse_variance"],
            "mae_mean": bairros["mae_mean"],
            "mae_variance": bairros["mae_variance"],
        },
        {
            "cenario": "fase2_skforecast_hexagonos_95",
            "best_model": hexagonos["best_model"],
            "selected_regions": hexagonos["selected_regions"],
            "achieved_coverage": hexagonos["achieved_coverage"],
            "mse_mean": hexagonos["mse_mean"],
            "mse_variance": hexagonos["mse_variance"],
            "mae_mean": hexagonos["mae_mean"],
            "mae_variance": hexagonos["mae_variance"],
        },
    ]
    best_global = min(rows, key=lambda item: item["mse_mean"])

    payload = {
        "bairros": bairros,
        "hexagonos": hexagonos,
        "comparison_rows": rows,
        "best_global_scenario": best_global["cenario"],
    }
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    lines = [
        "# Relatorio da Fase 2 com skforecast",
        "",
        "| Cenario | Unidades retidas | Cobertura | Melhor modelo | Media MSE | Variancia MSE | Media MAE | Variancia MAE |",
        "| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['cenario']} | {row['selected_regions']} | {row['achieved_coverage']:.4f} | "
            f"{row['best_model']} | {row['mse_mean']:.6f} | {row['mse_variance']:.6f} | "
            f"{row['mae_mean']:.6f} | {row['mae_variance']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Conclusao",
            "",
            f"- **Melhor cenario global na fase 2:** `{best_global['cenario']}`",
            f"- **Melhor modelo no cenario vencedor:** `{best_global['best_model']}`",
            "",
            "A fase 2 usa `skforecast` com estrategia recursiva (`ForecasterRecursive`) e mantem o mesmo corte espacial de 95% para garantir comparabilidade com a fase 1.",
        ]
    )

    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    print("Comparacao da fase 2 com skforecast gerada com sucesso.")
    print(f"Melhor cenario global: {best_global['cenario']}")
    print(f"Relatorio salvo em: {OUTPUT_MD_PATH}")


if __name__ == "__main__":
    main()
