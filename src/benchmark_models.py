import json
import os

import numpy as np
import pandas as pd

try:
    from sklearn.base import clone
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, PoissonRegressor, Ridge
    SKLEARN_AVAILABLE = True
except ImportError:
    clone = None
    HistGradientBoostingRegressor = None
    RandomForestRegressor = None
    LinearRegression = None
    PoissonRegressor = None
    Ridge = None
    SKLEARN_AVAILABLE = False

from predictor import evaluate_predictions, get_feature_columns, prepare_weekly_series, split_train_test


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_JSON_PATH = os.path.join(BASE_DIR, "data", "processed", "benchmark_cvli_hex_models.json")
BENCHMARK_MD_PATH = os.path.join(BASE_DIR, "BENCHMARK_CVLI_HEXAGONOS.md")
FORECAST_CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "cvli_hex_forecasts.csv")


def get_model_registry(selected_models=None):
    registry = {"naive_lag1": None}
    if SKLEARN_AVAILABLE:
        registry.update(
            {
                "linear_regression": LinearRegression(),
                "ridge": Ridge(alpha=1.0),
                "poisson": PoissonRegressor(alpha=0.1, max_iter=500),
                "random_forest": RandomForestRegressor(n_estimators=120, random_state=42, min_samples_leaf=2, n_jobs=1),
                "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42, max_depth=6, learning_rate=0.05),
            }
        )
    if selected_models is None:
        return registry
    return {name: registry[name] for name in selected_models if name in registry}


def _fit_predict_single_hex(model_name, model, df_hex, feature_cols):
    df_train, df_test = split_train_test(df_hex)
    if len(df_train) == 0 or len(df_test) == 0:
        return None

    if model_name == "naive_lag1":
        predictions = df_test["lag_1"].to_numpy()
    else:
        estimator = clone(model)
        estimator.fit(df_train[feature_cols], df_train["crimes"])
        predictions = estimator.predict(df_test[feature_cols])

    predictions = np.clip(np.asarray(predictions, dtype=float), 0, None)
    return {
        "y_true": df_test["crimes"].to_numpy(),
        "y_pred": predictions,
        "train_rows": len(df_train),
        "test_rows": len(df_test),
    }


def benchmark_models(df_hex, region_col="hex_id", lags=3, selected_models=None):
    feature_cols = get_feature_columns(lags=lags)
    registry = get_model_registry(selected_models=selected_models)
    regions = sorted(df_hex[region_col].unique().tolist())
    results = []

    for model_name, model in registry.items():
        all_true = []
        all_pred = []
        evaluated_regions = 0

        for region in regions:
            df_region = df_hex[df_hex[region_col] == region].copy()
            outcome = _fit_predict_single_hex(model_name, model, df_region, feature_cols)
            if outcome is None:
                continue
            all_true.extend(outcome["y_true"])
            all_pred.extend(outcome["y_pred"])
            evaluated_regions += 1

        if not all_true:
            continue

        metrics = evaluate_predictions(np.asarray(all_true), np.asarray(all_pred))
        results.append(
            {
                "model": model_name,
                "mse": float(metrics["mse"]),
                "mae": float(metrics["mae"]),
                "hexagons_evaluated": int(evaluated_regions),
            }
        )

    results = sorted(results, key=lambda item: item["mse"])
    return results


def train_best_model_and_forecast(df_hex, benchmark_results, region_col="hex_id", lags=3):
    if not benchmark_results:
        return None, pd.DataFrame()

    best_model_name = benchmark_results[0]["model"]
    return train_model_and_forecast(df_hex, best_model_name, region_col=region_col, lags=lags)


def train_model_and_forecast(df_hex, model_name, region_col="hex_id", lags=3):
    model_registry = get_model_registry()
    if model_name not in model_registry:
        model_name = "naive_lag1"
    model = model_registry[model_name]
    feature_cols = get_feature_columns(lags=lags)
    forecasts = []

    for region in sorted(df_hex[region_col].unique().tolist()):
        df_region = df_hex[df_hex[region_col] == region].copy().sort_values("semana")
        if len(df_region) < 8:
            continue

        latest = df_region.iloc[-1].copy()
        X_latest = latest[feature_cols].to_frame().T

        if model_name == "naive_lag1" or not SKLEARN_AVAILABLE:
            prediction = float(max(latest["lag_1"], 0))
        else:
            estimator = clone(model)
            estimator.fit(df_region[feature_cols], df_region["crimes"])
            prediction = float(max(estimator.predict(X_latest)[0], 0))

        forecasts.append(
            {
                region_col: int(region),
                "ultima_semana_observada": latest["semana"],
                "ultima_contagem": int(latest["crimes"]),
                "previsao_proxima_semana": prediction,
                "media_movel_4": float(latest.get("media_movel_4", 0.0)),
                "tendencia_1": float(latest.get("tendencia_1", 0.0)),
                "modelo_previsao": model_name,
            }
        )

    return model_name, pd.DataFrame(forecasts)


def save_benchmark_outputs(benchmark_results, best_model_name, forecasts_df):
    payload = {
        "ranking": benchmark_results,
        "best_model": best_model_name,
    }
    with open(BENCHMARK_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)

    if not forecasts_df.empty:
        forecasts_df.to_csv(FORECAST_CSV_PATH, index=False)

    lines = [
        "# Benchmark de Modelos para Séries Semanais por Hexágono (CVLI)",
        "",
        "## Ranking dos modelos",
        "",
        "| Modelo | MSE | MAE | Hexágonos avaliados |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for row in benchmark_results:
        lines.append(
            f"| {row['model']} | {row['mse']:.4f} | {row['mae']:.4f} | {row['hexagons_evaluated']} |"
        )

    lines.extend(
        [
            "",
            f"## Melhor modelo",
            "",
            f"- **Modelo vencedor:** `{best_model_name}`",
            f"- **Arquivo de previsões por hexágono:** `{FORECAST_CSV_PATH}`",
            "",
            "Cada previsão é individualizada por hexágono, usando a série semanal histórica daquela célula.",
        ]
    )

    with open(BENCHMARK_MD_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
