from experiment_protocol import (
    build_experiment_summary,
    build_weekly_series_for_regions,
    choose_best_model,
    evaluate_univariate_models_per_region,
    save_protocol_outputs,
    select_top_regions_by_coverage,
)
from spatial_utils import filter_cvli, filter_fortaleza_bbox, get_processed_data_path, load_processed_crimes


OUTPUT_PREFIX = "baseline_bairros_95"
DEFAULT_MODELS = ["naive_lag1", "linear_regression", "ridge", "poisson"]
DEFAULT_LAGS = 3
DEFAULT_TRAIN_SPLIT_DATE = "2025-12-31"


def main():
    data_path = get_processed_data_path()
    print(f"Carregando base para baseline por bairros: {data_path}")
    df = load_processed_crimes()
    df = filter_cvli(df)
    df["data"] = df["data"]
    df = filter_fortaleza_bbox(df)

    selected_bairros, coverage_df = select_top_regions_by_coverage(df, region_col="bairro", coverage=0.95)
    weekly_series = build_weekly_series_for_regions(df, region_col="bairro", selected_regions=selected_bairros, lags=DEFAULT_LAGS)

    experiment_results = evaluate_univariate_models_per_region(
        weekly_series,
        region_col="bairro",
        selected_models=DEFAULT_MODELS,
        lags=DEFAULT_LAGS,
        train_split_date=DEFAULT_TRAIN_SPLIT_DATE,
    )
    best_model_name = choose_best_model(experiment_results, metric="mse_mean")
    if best_model_name is None:
        raise RuntimeError("Nenhum modelo conseguiu gerar métricas válidas para o baseline por bairros.")

    best_payload = experiment_results[best_model_name]
    summary_payload = build_experiment_summary(
        experiment_name="baseline_bairros_95",
        region_col="bairro",
        selected_regions=selected_bairros,
        coverage_table=coverage_df,
        best_model_name=best_model_name,
        best_summary=best_payload["summary"],
        total_events=len(df),
        filtered_events=int(df[df["bairro"].isin(selected_bairros)].shape[0]),
        lags=DEFAULT_LAGS,
        train_split_date=DEFAULT_TRAIN_SPLIT_DATE,
        data_path=data_path,
        extra={"evaluated_models": DEFAULT_MODELS},
    )

    outputs = save_protocol_outputs(
        output_prefix=OUTPUT_PREFIX,
        weekly_series_df=weekly_series,
        coverage_df=coverage_df,
        best_region_metrics_df=best_payload["region_metrics"],
        best_predictions_df=best_payload["predictions"],
        summary_payload=summary_payload,
        report_title="Baseline por Bairros com Cobertura de 95%",
        region_label="bairros",
    )

    print("\n=== Baseline por bairros concluído ===")
    print(f"Bairros selecionados: {summary_payload['selected_regions']}")
    print(f"Cobertura atingida: {summary_payload['achieved_coverage']:.4f}")
    print(f"Melhor modelo: {best_model_name}")
    print(f"MSE médio: {summary_payload['mse_mean']}")
    print(f"Relatório salvo em: {outputs['report_path']}")


if __name__ == "__main__":
    main()
