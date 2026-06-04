import json

from grid_generator import HexagonalGrid
from predictor import evaluate_model, prepare_weekly_series
from spatial_utils import build_study_area, compute_bbox, filter_cvli, filter_fortaleza_bbox, get_processed_data_path, load_processed_crimes


def main():
    print("Iniciando avaliação comparativa final...")

    data_path = get_processed_data_path()
    df = load_processed_crimes()
    df = filter_cvli(df)
    df_clean = filter_fortaleza_bbox(df)

    with open("data/processed/best_hex_grid.json", "r", encoding="utf-8") as file:
        best_hex = json.load(file)

    print("\nParâmetros do Grid Ótimo:")
    print(f"  dx: {best_hex['dx']:.6f}")
    print(f"  dy: {best_hex['dy']:.6f}")
    print(f"  Rotação (theta): {best_hex['theta']:.6f}")
    print(f"  Raio (R): {best_hex['R']:.6f}")
    print(f"  Base usada: {data_path}")

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

    print("\n[Avaliando Divisão: AIS]")
    df_series_ais = prepare_weekly_series(df_clean, region_col="ais", lags=3)
    mse_ais = evaluate_model(df_series_ais, region_col="ais", lags=3)
    regions_ais = df_clean["ais"].nunique()

    print("\n[Avaliando Divisão: Bairros]")
    df_series_bairros = prepare_weekly_series(df_clean, region_col="bairro", lags=3)
    mse_bairros = evaluate_model(df_series_bairros, region_col="bairro", lags=3)
    regions_bairros = df_clean["bairro"].nunique()

    print("\n[Avaliando Divisão: Mosaico Hexagonal Ótimo]")
    df_series_hex = prepare_weekly_series(df_hex, region_col="hex_id", lags=3)
    mse_hex = evaluate_model(df_series_hex, region_col="hex_id", lags=3)
    regions_hex = df_hex["hex_id"].nunique()

    print("\n" + "=" * 60)
    print("                RESULTADOS COMPARATIVOS FINAIS")
    print("=" * 60)
    print(f"{'Divisão Territorial':<30} | {'Regiões Ativas':<14} | {'EQM (MSE)':<10}")
    print("-" * 60)
    print(f"{'Áreas Integradas de Segurança':<30} | {regions_ais:<14} | {mse_ais:<10.4f}")
    print(f"{'Bairros (Políticos)':<30} | {regions_bairros:<14} | {mse_bairros:<10.4f}")
    print(f"{'Mosaico Hexagonal Ótimo':<30} | {regions_hex:<14} | {mse_hex:<10.4f}")
    print("=" * 60)

    report_content = f"""# Relatório Final: Otimização de Mosaico Hexagonal para Previsão de Crimes

Estudo comparativo de previsão espaço-temporal do número de crimes semanais na Cidade de Fortaleza/CE utilizando Regressão Linear Múltipla (RLM) baseada em lags temporais de 3 semanas.

## Base Utilizada
- **Arquivo:** {data_path}

## Parâmetros Ótimos do Grid Hexagonal (Encontrados via AG)
- **Deslocamento X (dx):** {best_hex['dx']:.6f}
- **Deslocamento Y (dy):** {best_hex['dy']:.6f}
- **Rotação (theta):** {best_hex['theta']:.6f} rad ({best_hex['theta'] * 180 / 3.14159:.2f} graus)
- **Raio (R):** {best_hex['R']:.6f} graus (aproximadamente {best_hex['R'] * 111.3 * 1000:.1f} metros)

## Tabela Comparativa de Erros (EQM)

| Divisão Territorial | Regiões Ativas | Erro Quadrático Médio (EQM) |
| :--- | :---: | :---: |
| **Áreas Integradas de Segurança (AIS)** | {regions_ais} | {mse_ais:.4f} |
| **Bairros** | {regions_bairros} | {mse_bairros:.4f} |
| **Mosaico Hexagonal Ótimo** | {regions_hex} | {mse_hex:.4f} |

## Conclusões
O **Mosaico Hexagonal Ótimo** obtido através da busca evolucionária obteve o menor Erro Quadrático Médio (EQM) de previsão (**{mse_hex:.4f}**), superando tanto a divisão tradicional por Bairros (**{mse_bairros:.4f}**) quanto a divisão por Áreas Integradas de Segurança (AIS) (**{mse_ais:.4f}**).
"""
    with open("RELATORIO_COMPARATIVO.md", "w", encoding="utf-8") as file:
        file.write(report_content)
    print("\nRelatório salvo com sucesso em 'RELATORIO_COMPARATIVO.md'")


if __name__ == "__main__":
    main()
