from predictor import evaluate_model, prepare_weekly_series
from spatial_utils import get_processed_data_path, load_processed_crimes, filter_cvli


def main():
    data_path = get_processed_data_path()
    print(f"Carregando crimes de {data_path}...")
    df = load_processed_crimes()
    df = filter_cvli(df)

    print("\n[Avaliando Divisão: AIS]")
    df_ais = prepare_weekly_series(df, region_col="ais", lags=3)
    mse_ais = evaluate_model(df_ais, region_col="ais", lags=3)
    print(f"Número de linhas da série semanal: {len(df_ais)}")
    print(f"Erro Quadrático Médio (EQM) de previsão para AIS: {mse_ais:.4f}")

    print("\n[Avaliando Divisão: Bairros]")
    df_bairros = prepare_weekly_series(df, region_col="bairro", lags=3)
    mse_bairros = evaluate_model(df_bairros, region_col="bairro", lags=3)
    print(f"Número de linhas da série semanal: {len(df_bairros)}")
    print(f"Erro Quadrático Médio (EQM) de previsão para Bairros: {mse_bairros:.4f}")


if __name__ == "__main__":
    main()
