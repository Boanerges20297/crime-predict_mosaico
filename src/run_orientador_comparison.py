import json
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAIRROS_SUMMARY_PATH = os.path.join(BASE_DIR, "data", "processed", "baseline_bairros_95_summary.json")
HEX_SUMMARY_PATH = os.path.join(BASE_DIR, "data", "processed", "experimento_hexagonos_95_summary.json")
OUTPUT_JSON_PATH = os.path.join(BASE_DIR, "data", "processed", "orientador_comparison.json")
OUTPUT_MD_PATH = os.path.join(BASE_DIR, "RELATORIO_COMPARATIVO_ORIENTADOR.md")


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def compare_summaries(bairros, hexagonos):
    rows = [
        {
            "cenario": "bairros_95",
            "selected_regions": bairros["selected_regions"],
            "achieved_coverage": bairros["achieved_coverage"],
            "best_model": bairros["best_model"],
            "mse_mean": bairros["mse_mean"],
            "mse_variance": bairros["mse_variance"],
            "mae_mean": bairros["mae_mean"],
            "mae_variance": bairros["mae_variance"],
        },
        {
            "cenario": "hexagonos_95",
            "selected_regions": hexagonos["selected_regions"],
            "achieved_coverage": hexagonos["achieved_coverage"],
            "best_model": hexagonos["best_model"],
            "mse_mean": hexagonos["mse_mean"],
            "mse_variance": hexagonos["mse_variance"],
            "mae_mean": hexagonos["mae_mean"],
            "mae_variance": hexagonos["mae_variance"],
        },
    ]
    best_global = min(rows, key=lambda item: item["mse_mean"])
    return rows, best_global


def main():
    bairros = load_json(BAIRROS_SUMMARY_PATH)
    hexagonos = load_json(HEX_SUMMARY_PATH)
    rows, best_global = compare_summaries(bairros, hexagonos)

    payload = {
        "bairros": bairros,
        "hexagonos": hexagonos,
        "comparison_rows": rows,
        "best_global_scenario": best_global["cenario"],
    }
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    lines = [
        "# Relatório Comparativo do Protocolo do Orientador",
        "",
        "| Cenário | Unidades retidas | Cobertura | Melhor modelo | Média MSE | Variância MSE | Média MAE | Variância MAE |",
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
            "## Conclusão global",
            "",
            f"- **Melhor cenário pelo MSE médio:** `{best_global['cenario']}`",
            f"- **Modelo do cenário vencedor:** `{best_global['best_model']}`",
            "",
            "A comparação acima segue o mesmo protocolo para bairros e hexágonos: retenção das unidades que cobrem 95% dos eventos e previsão univariada por unidade espacial.",
        ]
    )

    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    print("Comparação final do orientador gerada com sucesso.")
    print(f"Melhor cenário global: {best_global['cenario']}")
    print(f"Relatório salvo em: {OUTPUT_MD_PATH}")


if __name__ == "__main__":
    main()
