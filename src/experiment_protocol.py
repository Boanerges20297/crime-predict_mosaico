import json
import os

import numpy as np
import pandas as pd
from sklearn.base import clone

from benchmark_models import get_model_registry
from predictor import evaluate_predictions, get_feature_columns, prepare_weekly_series, split_train_test


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def select_top_regions_by_coverage(df, region_col, coverage=0.95):
    if df.empty:
        return [], pd.DataFrame(columns=[region_col, "event_count", "event_share", "cumulative_share", "selected"])

    counts = (
        df.groupby(region_col)
        .size()
        .reset_index(name="event_count")
        .sort_values(["event_count", region_col], ascending=[False, True])
        .reset_index(drop=True)
    )
    total_events = int(counts["event_count"].sum())
    counts["event_share"] = counts["event_count"] / max(total_events, 1)
    counts["cumulative_share"] = counts["event_share"].cumsum()
    counts["selected"] = counts["cumulative_share"].shift(fill_value=0.0) < coverage

    selected_regions = counts.loc[counts["selected"], region_col].tolist()
    return selected_regions, counts


def restrict_to_selected_regions(df, region_col, selected_regions):
    if not selected_regions:
        return df.iloc[0:0].copy()
    return df[df[region_col].isin(selected_regions)].copy()


def build_weekly_series_for_regions(df, region_col, selected_regions, lags=3):
    df_selected = restrict_to_selected_regions(df, region_col, selected_regions)
    if df_selected.empty:
        return df_selected
    return prepare_weekly_series(df_selected, region_col=region_col, lags=lags)


def evaluate_univariate_models_per_region(
    df_series,
    region_col,
    selected_models=None,
    lags=3,
    train_split_date="2025-12-31",
):
    registry = get_model_registry(selected_models=selected_models)
    feature_cols = get_feature_columns(lags=lags)
    results = {}

    for model_name, model in registry.items():
        region_metrics = []
        region_predictions = []

        for region in sorted(df_series[region_col].unique().tolist()):
            df_region = df_series[df_series[region_col] == region].copy().sort_values("semana")
            df_train, df_test = split_train_test(df_region, train_split_date=train_split_date)
            if len(df_train) == 0 or len(df_test) == 0:
                region_metrics.append(
                    {
                        region_col: region,
                        "model": model_name,
                        "train_rows": int(len(df_train)),
                        "test_rows": int(len(df_test)),
                        "status": "skipped_empty_split",
                    }
                )
                continue

            if model_name == "naive_lag1":
                predictions = df_test["lag_1"].to_numpy()
            else:
                estimator = clone(model)
                estimator.fit(df_train[feature_cols], df_train["crimes"])
                predictions = estimator.predict(df_test[feature_cols])

            predictions = np.clip(np.asarray(predictions, dtype=float), 0, None)
            metrics = evaluate_predictions(df_test["crimes"].to_numpy(), predictions)
            region_metrics.append(
                {
                    region_col: region,
                    "model": model_name,
                    "train_rows": int(len(df_train)),
                    "test_rows": int(len(df_test)),
                    "mse": float(metrics["mse"]),
                    "mae": float(metrics["mae"]),
                    "status": "evaluated",
                }
            )
            region_predictions.append(
                pd.DataFrame(
                    {
                        region_col: region,
                        "model": model_name,
                        "semana": df_test["semana"].to_numpy(),
                        "y_true": df_test["crimes"].to_numpy(),
                        "y_pred": predictions,
                    }
                )
            )

        metrics_df = pd.DataFrame(region_metrics)
        predictions_df = pd.concat(region_predictions, ignore_index=True) if region_predictions else pd.DataFrame()
        summary = summarize_region_metrics(metrics_df)
        summary["model"] = model_name
        results[model_name] = {
            "region_metrics": metrics_df,
            "predictions": predictions_df,
            "summary": summary,
        }

    return results


def summarize_region_metrics(metrics_df):
    if metrics_df.empty or "status" not in metrics_df.columns:
        return {
            "regions_total": 0,
            "regions_evaluated": 0,
            "regions_skipped": 0,
            "mse_mean": None,
            "mse_variance": None,
            "mae_mean": None,
            "mae_variance": None,
        }

    evaluated = metrics_df[metrics_df["status"] == "evaluated"].copy()
    mse_series = evaluated["mse"] if "mse" in evaluated.columns else pd.Series(dtype=float)
    mae_series = evaluated["mae"] if "mae" in evaluated.columns else pd.Series(dtype=float)

    return {
        "regions_total": int(len(metrics_df)),
        "regions_evaluated": int(len(evaluated)),
        "regions_skipped": int(len(metrics_df) - len(evaluated)),
        "mse_mean": float(mse_series.mean()) if not mse_series.empty else None,
        "mse_variance": float(mse_series.var(ddof=0)) if not mse_series.empty else None,
        "mae_mean": float(mae_series.mean()) if not mae_series.empty else None,
        "mae_variance": float(mae_series.var(ddof=0)) if not mae_series.empty else None,
    }


def choose_best_model(experiment_results, metric="mse_mean"):
    ranked = []
    for model_name, payload in experiment_results.items():
        metric_value = payload["summary"].get(metric)
        if metric_value is None or pd.isna(metric_value):
            continue
        ranked.append((metric_value, model_name))

    ranked.sort(key=lambda item: item[0])
    return ranked[0][1] if ranked else None


def build_experiment_summary(
    *,
    experiment_name,
    region_col,
    selected_regions,
    coverage_table,
    best_model_name,
    best_summary,
    total_events,
    filtered_events,
    lags,
    train_split_date,
    data_path,
    extra=None,
):
    achieved_coverage = float(filtered_events / max(total_events, 1)) if total_events else 0.0
    summary = {
        "experiment_name": experiment_name,
        "region_col": region_col,
        "coverage_threshold": 0.95,
        "selected_regions": int(len(selected_regions)),
        "total_events": int(total_events),
        "filtered_events": int(filtered_events),
        "achieved_coverage": achieved_coverage,
        "best_model": best_model_name,
        "lags": int(lags),
        "train_split_date": str(train_split_date),
        "data_path": data_path,
        "coverage_table_preview": coverage_table.head(10).to_dict(orient="records"),
    }
    summary.update(best_summary or {})
    if extra:
        summary.update(extra)
    return summary


def save_protocol_outputs(
    *,
    output_prefix,
    weekly_series_df,
    coverage_df,
    best_region_metrics_df,
    best_predictions_df,
    summary_payload,
    report_title,
    region_label,
):
    weekly_path = os.path.join(BASE_DIR, "data", "processed", f"{output_prefix}_weekly_series.csv")
    coverage_path = os.path.join(BASE_DIR, "data", "processed", f"{output_prefix}_coverage.csv")
    metrics_path = os.path.join(BASE_DIR, "data", "processed", f"{output_prefix}_metrics.csv")
    predictions_path = os.path.join(BASE_DIR, "data", "processed", f"{output_prefix}_predictions.csv")
    summary_path = os.path.join(BASE_DIR, "data", "processed", f"{output_prefix}_summary.json")
    report_path = os.path.join(BASE_DIR, f"{output_prefix.upper()}.md")

    weekly_series_df.to_csv(weekly_path, index=False)
    coverage_df.to_csv(coverage_path, index=False)
    best_region_metrics_df.to_csv(metrics_path, index=False)
    if not best_predictions_df.empty:
        best_predictions_df.to_csv(predictions_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary_payload, file, indent=2, ensure_ascii=False, default=str)

    lines = [
        f"# {report_title}",
        "",
        f"- **Modelo selecionado:** `{summary_payload.get('best_model')}`",
        f"- **Unidades selecionadas ({region_label}):** {summary_payload.get('selected_regions')}",
        f"- **Eventos totais:** {summary_payload.get('total_events')}",
        f"- **Eventos retidos:** {summary_payload.get('filtered_events')}",
        f"- **Cobertura atingida:** {summary_payload.get('achieved_coverage', 0.0):.4f}",
        f"- **Média do MSE:** {summary_payload.get('mse_mean')}",
        f"- **Variância do MSE:** {summary_payload.get('mse_variance')}",
        f"- **Média do MAE:** {summary_payload.get('mae_mean')}",
        f"- **Variância do MAE:** {summary_payload.get('mae_variance')}",
        "",
        "## Arquivos gerados",
        "",
        f"- `data/processed/{output_prefix}_weekly_series.csv`",
        f"- `data/processed/{output_prefix}_coverage.csv`",
        f"- `data/processed/{output_prefix}_metrics.csv`",
        f"- `data/processed/{output_prefix}_predictions.csv`",
        f"- `data/processed/{output_prefix}_summary.json`",
    ]

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    return {
        "weekly_path": weekly_path,
        "coverage_path": coverage_path,
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
        "summary_path": summary_path,
        "report_path": report_path,
    }
