import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error
except ImportError:
    LinearRegression = None

    def mean_squared_error(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.size == 0:
            return np.nan
        diff = y_true - y_pred
        return float(np.mean(diff * diff))

    def mean_absolute_error(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.size == 0:
            return np.nan
        return float(np.mean(np.abs(y_true - y_pred)))


def prepare_weekly_series(df, region_col, lags=3):
    """
    Agrupa os crimes semanalmente e gera variáveis de lag por região.
    """
    df = df.copy()
    df["data"] = pd.to_datetime(df["data"])
    df["semana"] = df["data"].dt.to_period("W").dt.start_time

    weekly_counts = df.groupby(["semana", region_col]).size().reset_index(name="crimes")

    semanas_unicas = pd.date_range(
        start=weekly_counts["semana"].min(),
        end=weekly_counts["semana"].max(),
        freq="W-MON",
    )
    regioes_unicas = weekly_counts[region_col].unique()

    multi_idx = pd.MultiIndex.from_product([semanas_unicas, regioes_unicas], names=["semana", region_col])
    grid_df = pd.DataFrame(index=multi_idx).reset_index()

    merged = pd.merge(grid_df, weekly_counts, on=["semana", region_col], how="left")
    merged["crimes"] = merged["crimes"].fillna(0).astype(int)
    merged = merged.sort_values(by=[region_col, "semana"]).reset_index(drop=True)

    for lag in range(1, lags + 1):
        merged[f"lag_{lag}"] = merged.groupby(region_col)["crimes"].shift(lag)

    merged["media_movel_4"] = (
        merged.groupby(region_col)["crimes"]
        .transform(lambda series: series.shift(1).rolling(window=4, min_periods=1).mean())
    )
    merged["tendencia_1"] = merged["lag_1"] - merged["lag_2"]
    merged["semana_ano"] = merged["semana"].dt.isocalendar().week.astype(int)
    merged["mes"] = merged["semana"].dt.month.astype(int)

    merged = merged.dropna().reset_index(drop=True)
    return merged


def get_feature_columns(lags=3):
    return [f"lag_{lag}" for lag in range(1, lags + 1)] + ["media_movel_4", "tendencia_1", "semana_ano", "mes"]


def split_train_test(df_model, train_split_date="2025-12-31"):
    train_split_date = pd.to_datetime(train_split_date)
    train_mask = df_model["semana"] <= train_split_date
    test_mask = df_model["semana"] > train_split_date
    return df_model[train_mask].copy(), df_model[test_mask].copy()


def evaluate_model(df_model, region_col, lags=3, train_split_date="2025-12-31"):
    if LinearRegression is None:
        return np.nan

    df_train, df_test = split_train_test(df_model, train_split_date=train_split_date)
    if len(df_train) == 0 or len(df_test) == 0:
        return np.nan

    features = get_feature_columns(lags=lags)
    model = LinearRegression()
    model.fit(df_train[features], df_train["crimes"])
    predictions = np.clip(model.predict(df_test[features]), 0, None)
    return mean_squared_error(df_test["crimes"], predictions)


def evaluate_predictions(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }
