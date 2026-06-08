from skforecast.recursive import ForecasterRecursive

from skforecast_protocol import run_skforecast_phase
from spatial_utils import filter_cvli, filter_fortaleza_bbox, get_processed_data_path, load_processed_crimes


OUTPUT_PREFIX = "fase2_skforecast_bairros_95"
SELECTED_MODELS = ["linear_regression", "ridge", "poisson", "hist_gradient_boosting"]
LAGS = 3
TRAIN_SPLIT_DATE = "2025-12-31"


def main():
    data_path = get_processed_data_path()
    print(f"Carregando base para fase 2 skforecast em bairros: {data_path}")
    df = load_processed_crimes()
    df = filter_cvli(df)
    df = filter_fortaleza_bbox(df)

    result = run_skforecast_phase(
        df=df,
        region_col="bairro",
        output_prefix=OUTPUT_PREFIX,
        report_title="Fase 2 com skforecast em Bairros (95%)",
        region_label="bairros",
        data_path=data_path,
        forecaster_cls=ForecasterRecursive,
        selected_models=SELECTED_MODELS,
        lags=LAGS,
        train_split_date=TRAIN_SPLIT_DATE,
        extra_summary={
            "framework": "skforecast",
            "skforecast_strategy": "ForecasterRecursive",
            "evaluated_models": SELECTED_MODELS,
        },
    )

    print("\n=== Fase 2 skforecast em bairros concluida ===")
    print(f"Melhor modelo: {result['best_model']}")
    print(f"MSE medio: {result['summary']['mse_mean']}")
    print(f"Relatorio salvo em: {result['outputs']['report_path']}")


if __name__ == "__main__":
    main()
