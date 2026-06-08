import json
import os

import numpy as np
import pandas as pd
from sklearn.base import clone

from benchmark_models import get_model_registry
from experiment_protocol import (
    build_experiment_summary,
    build_weekly_series_for_regions,
    choose_best_model,
    restrict_to_selected_regions,
    save_protocol_outputs,
    select_top_regions_by_coverage,
)
from predictor import evaluate_predictions


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def prepare_region_series_dict(df, region_col):
    temp = df.copy()
    temp["data"] = pd.to_datetime(temp["data"], errors="coerce")
    temp["semana"] = temp["data"].dt.to_period("W").dt.start_time

    weekly_counts = temp.groupby(["semana", region_col]).size().reset_index(name="crimes")
    min_week = weekly_counts["semana"].min()
    max_week = weekly_counts["semana"].max()
    full_index = pd.date_range(start=min_week, end=max_week, freq="W-MON")

    series_dict = {}
    for region in sorted(weekly_counts[region_col].unique().tolist()):
        region_weekly = weekly_counts[weekly_counts[region_col] == region][["semana", "crimes"]].copy()
        region_series = (
            region_weekly.set_index("semana")
            .reindex(full_index)
            .fillna(0)
            .rename_axis("semana")["crimes"]
            .astype(float)
        )
        region_series.name = str(region)
        series_dict[region] = region_series

    return series_dict


def split_series_train_test(region_series, train_split_date):
    split_ts = pd.to_datetime(train_split_date)
    train = region_series.loc[region_series.index <= split_ts].copy()
    test = region_series.loc[region_series.index > split_ts].copy()
    return train, test


def evaluate_skforecast_models_per_region(
    *,
    series_dict,
    region_col,
    forecaster_cls,
    selected_models,
    lags,
    train_split_date,
):
    registry = get_model_registry(selected_models=selected_models)
    registry = {name: model for name, model in registry.items() if name != "naive_lag1"}
    results = {}

    for model_name, estimator in registry.items():
        region_metrics = []
        region_predictions = []
        next_step_rows = []

        for region, region_series in series_dict.items():
            train, test = split_series_train_test(region_series, train_split_date=train_split_date)
            if len(train) <= lags or len(test) == 0:
                region_metrics.append(
                    {
                        region_col: region,
                        "model": model_name,
                        "train_rows": int(len(train)),
                        "test_rows": int(len(test)),
                        "status": "skipped_empty_split",
                    }
                )
                continue

            forecaster = forecaster_cls(
                estimator=clone(estimator),
                lags=lags,
            )
            forecaster.fit(y=train)
            predictions = forecaster.predict(steps=len(test), suppress_warnings=True)
            predictions = np.clip(np.asarray(predictions, dtype=float), 0, None)

            metrics = evaluate_predictions(test.to_numpy(), predictions)
            region_metrics.append(
                {
                    region_col: region,
                    "model": model_name,
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
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
                        "semana": test.index,
                        "y_true": test.to_numpy(),
                        "y_pred": predictions,
                    }
                )
            )

            full_forecaster = forecaster_cls(
                estimator=clone(estimator),
                lags=lags,
            )
            full_forecaster.fit(y=region_series)
            next_pred = float(max(full_forecaster.predict(steps=1, suppress_warnings=True).iloc[0], 0.0))
            next_step_rows.append(
                {
                    region_col: region,
                    "model": model_name,
                    "ultima_semana_observada": region_series.index.max(),
                    "ultima_contagem": float(region_series.iloc[-1]),
                    "previsao_proxima_semana": next_pred,
                }
            )

        metrics_df = pd.DataFrame(region_metrics)
        predictions_df = pd.concat(region_predictions, ignore_index=True) if region_predictions else pd.DataFrame()
        next_step_df = pd.DataFrame(next_step_rows)
        summary = summarize_metrics(metrics_df)
        summary["model"] = model_name
        results[model_name] = {
            "region_metrics": metrics_df,
            "predictions": predictions_df,
            "next_step_forecasts": next_step_df,
            "summary": summary,
        }

    return results


def summarize_metrics(metrics_df):
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
    mse = evaluated["mse"] if "mse" in evaluated.columns else pd.Series(dtype=float)
    mae = evaluated["mae"] if "mae" in evaluated.columns else pd.Series(dtype=float)
    return {
        "regions_total": int(len(metrics_df)),
        "regions_evaluated": int(len(evaluated)),
        "regions_skipped": int(len(metrics_df) - len(evaluated)),
        "mse_mean": float(mse.mean()) if not mse.empty else None,
        "mse_variance": float(mse.var(ddof=0)) if not mse.empty else None,
        "mae_mean": float(mae.mean()) if not mae.empty else None,
        "mae_variance": float(mae.var(ddof=0)) if not mae.empty else None,
    }


def save_skforecast_outputs(
    *,
    output_prefix,
    weekly_series_df,
    coverage_df,
    best_region_metrics_df,
    best_predictions_df,
    best_next_step_df,
    summary_payload,
    report_title,
    region_label,
):
    outputs = save_protocol_outputs(
        output_prefix=output_prefix,
        weekly_series_df=weekly_series_df,
        coverage_df=coverage_df,
        best_region_metrics_df=best_region_metrics_df,
        best_predictions_df=best_predictions_df,
        summary_payload=summary_payload,
        report_title=report_title,
        region_label=region_label,
    )

    next_step_path = os.path.join(BASE_DIR, "data", "processed", f"{output_prefix}_next_step_forecasts.csv")
    best_next_step_df.to_csv(next_step_path, index=False)

    report_path = outputs["report_path"]
    with open(report_path, "a", encoding="utf-8") as file:
        file.write("\n")
        file.write(f"- `data/processed/{output_prefix}_next_step_forecasts.csv`\n")

    outputs["next_step_path"] = next_step_path
    return outputs


def run_skforecast_phase(
    *,
    df,
    region_col,
    output_prefix,
    report_title,
    region_label,
    data_path,
    forecaster_cls,
    selected_models,
    lags,
    train_split_date,
    extra_summary=None,
):
    selected_regions, coverage_df = select_top_regions_by_coverage(df, region_col=region_col, coverage=0.95)
    df_selected = restrict_to_selected_regions(df, region_col=region_col, selected_regions=selected_regions)
    weekly_series = build_weekly_series_for_regions(df, region_col=region_col, selected_regions=selected_regions, lags=lags)
    series_dict = prepare_region_series_dict(df_selected, region_col=region_col)

    experiment_results = evaluate_skforecast_models_per_region(
        series_dict=series_dict,
        region_col=region_col,
        forecaster_cls=forecaster_cls,
        selected_models=selected_models,
        lags=lags,
        train_split_date=train_split_date,
    )
    best_model_name = choose_best_model(experiment_results, metric="mse_mean")
    if best_model_name is None:
        raise RuntimeError(f"Nenhum modelo skforecast gerou métricas válidas para {output_prefix}.")

    best_payload = experiment_results[best_model_name]
    summary_payload = build_experiment_summary(
        experiment_name=output_prefix,
        region_col=region_col,
        selected_regions=selected_regions,
        coverage_table=coverage_df,
        best_model_name=best_model_name,
        best_summary=best_payload["summary"],
        total_events=len(df),
        filtered_events=len(df_selected),
        lags=lags,
        train_split_date=train_split_date,
        data_path=data_path,
        extra=extra_summary,
    )

    outputs = save_skforecast_outputs(
        output_prefix=output_prefix,
        weekly_series_df=weekly_series,
        coverage_df=coverage_df,
        best_region_metrics_df=best_payload["region_metrics"],
        best_predictions_df=best_payload["predictions"],
        best_next_step_df=best_payload["next_step_forecasts"],
        summary_payload=summary_payload,
        report_title=report_title,
        region_label=region_label,
    )

    summary_path = os.path.join(BASE_DIR, "data", "processed", f"{output_prefix}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary_payload, file, indent=2, ensure_ascii=False, default=str)

    return {
        "best_model": best_model_name,
        "summary": summary_payload,
        "outputs": outputs,
    }
