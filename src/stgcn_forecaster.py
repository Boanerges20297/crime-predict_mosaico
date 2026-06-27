import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = None
except ImportError as exc:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(exc)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STGCN_TESTS_JSON_PATH = os.path.join(BASE_DIR, "data", "processed", "stgcn_precision_tests.json")
STGCN_TESTS_MD_PATH = os.path.join(BASE_DIR, "STGCN_PRECISAO_TESTES.md")
STGCN_FORECASTS_CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "stgcn_hex_forecasts.csv")
STGCN_METADATA_JSON_PATH = os.path.join(BASE_DIR, "data", "processed", "stgcn_hex_metadata.json")

DEFAULT_SEQ_LEN = 12
DEFAULT_HIDDEN_DIM = 48
DEFAULT_EPOCHS = 140
DEFAULT_LEARNING_RATE = 0.004
DEFAULT_WEIGHT_DECAY = 2e-4
DEFAULT_BATCH_SIZE = 12
DEFAULT_GRAPH_NEIGHBORS = 6
DEFAULT_DROPOUT = 0.10
DEFAULT_PATIENCE = 18
DEFAULT_VALIDATION_RATIO = 0.20
DEFAULT_ENSEMBLE_SIZE = 2
DEFAULT_SEEDS = (42, 99, 123)
DEFAULT_DASHBOARD_SEQ_LEN = 8
DEFAULT_DASHBOARD_HIDDEN_DIM = 24
DEFAULT_DASHBOARD_EPOCHS = 28
DEFAULT_DASHBOARD_ENSEMBLE_SIZE = 1
DEFAULT_DASHBOARD_PATIENCE = 6


def is_stgcn_available():
    return TORCH_AVAILABLE


def get_stgcn_import_error():
    return TORCH_IMPORT_ERROR


if TORCH_AVAILABLE:
    class GraphTemporalBlock(nn.Module):
        def __init__(self, in_channels, out_channels, dropout):
            super().__init__()
            self.temporal = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
            self.spatial = nn.Linear(out_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm2d(out_channels)
            self.dropout = nn.Dropout(dropout)
            self.residual = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
            )

        def forward(self, x, adjacency):
            residual = self.residual(x)
            x = self.temporal(x)
            x = torch.einsum("bctn,nm->bctm", x, adjacency)
            x = x.permute(0, 2, 3, 1)
            x = self.spatial(x)
            x = x.permute(0, 3, 1, 2)
            x = self.norm(x)
            x = F.gelu(x + residual)
            return self.dropout(x)


    class STGCNPoisson(nn.Module):
        def __init__(self, input_dim, hidden_dim=DEFAULT_HIDDEN_DIM, dropout=DEFAULT_DROPOUT):
            super().__init__()
            self.block1 = GraphTemporalBlock(input_dim, hidden_dim, dropout)
            self.block2 = GraphTemporalBlock(hidden_dim, hidden_dim, dropout)
            self.block3 = GraphTemporalBlock(hidden_dim, hidden_dim, dropout)
            self.readout = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x, adjacency):
            x = x.permute(0, 3, 1, 2)
            x = self.block1(x, adjacency)
            x = self.block2(x, adjacency)
            x = self.block3(x, adjacency)
            x = x[:, :, -1, :].permute(0, 2, 1)
            x = self.readout(x).squeeze(-1)
            return F.softplus(x) + 1e-4
else:
    class STGCNPoisson:
        pass


def _build_adjacency(region_ids, grid, neighbors=DEFAULT_GRAPH_NEIGHBORS):
    node_count = len(region_ids)
    if node_count == 0:
        return np.zeros((0, 0), dtype=np.float32)

    if grid is None or not hasattr(grid, "display_hexagons"):
        return np.eye(node_count, dtype=np.float32)

    centroids = []
    for region_id in region_ids:
        geometry = grid.display_hexagons[int(region_id)]
        centroid = geometry.centroid
        centroids.append([float(centroid.x), float(centroid.y)])

    coords = np.asarray(centroids, dtype=np.float32)
    if node_count == 1:
        return np.eye(1, dtype=np.float32)

    n_neighbors = min(neighbors + 1, node_count)
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(coords)
    distances, indices = knn.kneighbors(coords)

    adjacency = np.zeros((node_count, node_count), dtype=np.float32)
    positive = distances[distances > 0]
    sigma = float(np.median(positive)) if positive.size else 1.0
    sigma = max(sigma, 1e-6)

    for source in range(node_count):
        adjacency[source, source] = 1.0
        for distance, target in zip(distances[source][1:], indices[source][1:]):
            weight = math.exp(-(float(distance) ** 2) / (2 * sigma ** 2))
            adjacency[source, target] = max(adjacency[source, target], weight)
            adjacency[target, source] = max(adjacency[target, source], weight)

    adjacency += np.eye(node_count, dtype=np.float32)
    degrees = adjacency.sum(axis=1)
    degrees[degrees == 0] = 1.0
    inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    normalized = inv_sqrt @ adjacency @ inv_sqrt
    return normalized.astype(np.float32)


def _build_feature_tensor(values, weeks):
    log_counts = np.log1p(values)
    sqrt_counts = np.sqrt(values)
    active_flag = (values > 0).astype(np.float32)
    moving_mean = pd.DataFrame(values).rolling(window=4, min_periods=1).mean().to_numpy(dtype=np.float32)
    moving_mean = np.log1p(moving_mean)
    week_numbers = np.asarray(pd.DatetimeIndex(weeks).isocalendar().week, dtype=np.float32)
    sin_week = np.sin(2 * np.pi * week_numbers / 52.0)[:, None]
    cos_week = np.cos(2 * np.pi * week_numbers / 52.0)[:, None]
    sin_week = np.repeat(sin_week, values.shape[1], axis=1)
    cos_week = np.repeat(cos_week, values.shape[1], axis=1)
    stacked = np.stack(
        [log_counts, sqrt_counts, active_flag, moving_mean, sin_week, cos_week],
        axis=-1,
    )
    return stacked.astype(np.float32)


def _prepare_sequences(df_model, region_col, train_split_date, seq_len, validation_ratio=DEFAULT_VALIDATION_RATIO):
    pivot = (
        df_model.pivot(index="semana", columns=region_col, values="crimes")
        .sort_index()
        .fillna(0.0)
    )
    if pivot.empty or len(pivot.index) <= seq_len:
        return None

    weeks = pd.to_datetime(pivot.index)
    region_ids = [int(region_id) for region_id in pivot.columns.tolist()]
    values = pivot.to_numpy(dtype=np.float32)
    features = _build_feature_tensor(values, weeks)
    train_split_date = pd.to_datetime(train_split_date)

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for target_idx in range(seq_len, len(values)):
        x_window = features[target_idx - seq_len:target_idx]
        y_target = values[target_idx]
        target_week = weeks[target_idx]
        if target_week <= train_split_date:
            train_x.append(x_window)
            train_y.append(y_target)
        else:
            test_x.append(x_window)
            test_y.append(y_target)

    all_x = [features[target_idx - seq_len:target_idx] for target_idx in range(seq_len, len(values))]
    all_y = [values[target_idx] for target_idx in range(seq_len, len(values))]
    latest_window = features[-seq_len:]

    train_x = np.asarray(train_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.float32)
    if train_x.size == 0:
        return None

    val_size = int(max(1, round(len(train_x) * validation_ratio))) if len(train_x) > 4 else 0
    if val_size >= len(train_x):
        val_size = max(0, len(train_x) - 1)

    if val_size > 0:
        fit_x = train_x[:-val_size]
        fit_y = train_y[:-val_size]
        val_x = train_x[-val_size:]
        val_y = train_y[-val_size:]
    else:
        fit_x = train_x
        fit_y = train_y
        val_x = np.asarray([], dtype=np.float32)
        val_y = np.asarray([], dtype=np.float32)

    return {
        "region_ids": region_ids,
        "weeks": weeks,
        "pivot": pivot,
        "fit_x": fit_x,
        "fit_y": fit_y,
        "val_x": val_x,
        "val_y": val_y,
        "train_x": train_x,
        "train_y": train_y,
        "test_x": np.asarray(test_x, dtype=np.float32),
        "test_y": np.asarray(test_y, dtype=np.float32),
        "all_x": np.asarray(all_x, dtype=np.float32),
        "all_y": np.asarray(all_y, dtype=np.float32),
        "latest_window": latest_window.astype(np.float32),
        "input_dim": features.shape[-1],
    }


def _seed_everything(seed):
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _build_model(input_dim, hidden_dim, dropout, learning_rate, weight_decay, device):
    model = STGCNPoisson(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=6,
        min_lr=5e-5,
    )
    criterion = nn.PoissonNLLLoss(log_input=False, full=False)
    return model, optimizer, scheduler, criterion


def _evaluate_loss(model, adjacency_tensor, criterion, x_values, y_values, device):
    if x_values.size == 0 or y_values.size == 0:
        return None
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_values, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y_values, dtype=torch.float32, device=device)
        predictions = model(x_tensor, adjacency_tensor)
        loss = criterion(predictions, y_tensor)
    return float(loss.item())


def _fit_single_model(
    train_x,
    train_y,
    val_x,
    val_y,
    adjacency,
    input_dim,
    seed,
    hidden_dim,
    dropout,
    epochs,
    learning_rate,
    weight_decay,
    batch_size,
    patience,
    progress_prefix=None,
):
    if not TORCH_AVAILABLE:
        raise RuntimeError(f"PyTorch indisponível para ST-GCN: {TORCH_IMPORT_ERROR}")
    if train_x.size == 0 or train_y.size == 0:
        raise RuntimeError("Dados insuficientes para treinar o ST-GCN.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _seed_everything(seed)

    adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32, device=device)
    model, optimizer, scheduler, criterion = _build_model(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
    )

    x_tensor = torch.tensor(train_x, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(train_y, dtype=torch.float32, device=device)

    best_state = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    sample_count = x_tensor.shape[0]

    progress_label = progress_prefix or "ST-GCN"

    for epoch_idx in range(epochs):
        permutation = torch.randperm(sample_count, device=device)
        model.train()
        for start in range(0, sample_count, batch_size):
            batch_idx = permutation[start:start + batch_size]
            batch_x = x_tensor[batch_idx]
            batch_y = y_tensor[batch_idx]

            optimizer.zero_grad()
            prediction = model(batch_x, adjacency_tensor)
            loss = criterion(prediction, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_loss = _evaluate_loss(model, adjacency_tensor, criterion, val_x, val_y, device)
        train_loss = _evaluate_loss(model, adjacency_tensor, criterion, train_x, train_y, device)
        monitored_loss = val_loss if val_loss is not None else train_loss
        scheduler.step(monitored_loss)

        train_loss_text = f"{train_loss:.6f}" if train_loss is not None else "n/a"
        val_loss_text = f"{val_loss:.6f}" if val_loss is not None else "n/a"
        best_loss_text = f"{best_val_loss:.6f}" if math.isfinite(best_val_loss) else "n/a"
        print(
            f"[{progress_label}] epoca {epoch_idx + 1}/{epochs} | "
            f"train_loss={train_loss_text} | val_loss={val_loss_text} | "
            f"best={best_loss_text} | paciencia={epochs_without_improvement}/{patience}"
        )

        if monitored_loss < best_val_loss:
            best_val_loss = monitored_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
            print(f"[{progress_label}] nova melhor perda monitorada: {best_val_loss:.6f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[{progress_label}] early stopping na epoca {epoch_idx + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, adjacency_tensor, device, best_val_loss


def _predict(model, adjacency_tensor, device, x_values):
    if x_values.size == 0:
        return np.asarray([], dtype=np.float32)

    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_values, dtype=torch.float32, device=device)
        predictions = model(x_tensor, adjacency_tensor).cpu().numpy()
    return np.clip(predictions, 0.0, None)


def _ensemble_predict(
    train_x,
    train_y,
    val_x,
    val_y,
    adjacency,
    input_dim,
    predict_x,
    ensemble_size,
    hidden_dim,
    dropout,
    epochs,
    learning_rate,
    weight_decay,
    batch_size,
    patience,
    progress_label=None,
):
    predictions = []
    losses = []
    selected_seeds = DEFAULT_SEEDS[:ensemble_size]
    ensemble_label = progress_label or "ST-GCN"
    for ensemble_idx, seed in enumerate(selected_seeds, start=1):
        seed_label = f"{ensemble_label} | semente {ensemble_idx}/{len(selected_seeds)} ({seed})"
        print(f"[{seed_label}] iniciando treino")
        model, adjacency_tensor, device, best_loss = _fit_single_model(
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            adjacency=adjacency,
            input_dim=input_dim,
            seed=seed,
            hidden_dim=hidden_dim,
            dropout=dropout,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            patience=patience,
            progress_prefix=seed_label,
        )
        predictions.append(_predict(model, adjacency_tensor, device, predict_x))
        losses.append(best_loss)
        print(f"[{seed_label}] concluido | best_loss={best_loss:.6f}")
    return np.mean(predictions, axis=0), float(np.mean(losses))


def run_stgcn_pipeline(
    df_model,
    region_col="hex_id",
    grid=None,
    train_split_date="2025-12-31",
    seq_len=DEFAULT_SEQ_LEN,
    ensemble_size=DEFAULT_ENSEMBLE_SIZE,
    hidden_dim=DEFAULT_HIDDEN_DIM,
    dropout=DEFAULT_DROPOUT,
    epochs=DEFAULT_EPOCHS,
    learning_rate=DEFAULT_LEARNING_RATE,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    batch_size=DEFAULT_BATCH_SIZE,
    patience=DEFAULT_PATIENCE,
):
    if not TORCH_AVAILABLE:
        raise RuntimeError(f"PyTorch indisponível para ST-GCN: {TORCH_IMPORT_ERROR}")

    print("[ST-GCN] preparando sequencias temporais")
    prepared = _prepare_sequences(
        df_model,
        region_col=region_col,
        train_split_date=train_split_date,
        seq_len=seq_len,
    )
    if prepared is None:
        raise RuntimeError("Série histórica insuficiente para montar janelas do ST-GCN.")

    region_ids = prepared["region_ids"]
    print(
        f"[ST-GCN] janelas prontas | nos={len(region_ids)} | "
        f"fit={len(prepared['fit_x'])} | val={len(prepared['val_x'])} | "
        f"test={len(prepared['test_x'])} | input_dim={prepared['input_dim']}"
    )
    adjacency = _build_adjacency(region_ids, grid=grid)

    if prepared["fit_x"].size == 0 or prepared["test_x"].size == 0:
        raise RuntimeError("Divisão temporal insuficiente para avaliar o ST-GCN.")

    print("[ST-GCN] iniciando treino para avaliacao temporal")
    eval_predictions, best_val_loss = _ensemble_predict(
        train_x=prepared["fit_x"],
        train_y=prepared["fit_y"],
        val_x=prepared["val_x"],
        val_y=prepared["val_y"],
        adjacency=adjacency,
        input_dim=prepared["input_dim"],
        predict_x=prepared["test_x"],
        ensemble_size=ensemble_size,
        hidden_dim=hidden_dim,
        dropout=dropout,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        patience=patience,
        progress_label="ST-GCN avaliacao",
    )

    mse = float(mean_squared_error(prepared["test_y"].reshape(-1), eval_predictions.reshape(-1)))
    mae = float(mean_absolute_error(prepared["test_y"].reshape(-1), eval_predictions.reshape(-1)))
    print(f"[ST-GCN] avaliacao concluida | mse={mse:.6f} | mae={mae:.6f}")

    print("[ST-GCN] iniciando treino para previsao operacional t+1")
    next_step_prediction, _ = _ensemble_predict(
        train_x=prepared["train_x"],
        train_y=prepared["train_y"],
        val_x=prepared["val_x"],
        val_y=prepared["val_y"],
        adjacency=adjacency,
        input_dim=prepared["input_dim"],
        predict_x=prepared["latest_window"][None, ...],
        ensemble_size=ensemble_size,
        hidden_dim=hidden_dim,
        dropout=dropout,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        patience=patience,
        progress_label="ST-GCN previsao_t+1",
    )
    next_step_prediction = next_step_prediction[0]
    print("[ST-GCN] previsao operacional concluida")

    latest_rows = (
        df_model.sort_values("semana")
        .groupby(region_col)
        .tail(1)
        .set_index(region_col)
    )

    forecasts = []
    for node_position, region_id in enumerate(region_ids):
        latest = latest_rows.loc[region_id]
        forecasts.append(
            {
                region_col: int(region_id),
                "ultima_semana_observada": latest["semana"],
                "ultima_contagem": int(latest["crimes"]),
                "previsao_proxima_semana": float(next_step_prediction[node_position]),
                "media_movel_4": float(latest.get("media_movel_4", 0.0)),
                "tendencia_1": float(latest.get("tendencia_1", 0.0)),
                "modelo_previsao": "stgcn_poisson_nll",
            }
        )

    return {
        "model_name": "stgcn_poisson_nll",
        "mse": mse,
        "mae": mae,
        "forecasts_df": pd.DataFrame(forecasts),
        "node_count": len(region_ids),
        "seq_len": seq_len,
        "best_validation_loss": best_val_loss,
        "ensemble_size": ensemble_size,
        "input_dim": prepared["input_dim"],
        "hidden_dim": hidden_dim,
        "epochs": epochs,
    }


def save_stgcn_precision_report(payload):
    os.makedirs(os.path.dirname(STGCN_TESTS_JSON_PATH), exist_ok=True)
    with open(STGCN_TESTS_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)

    lines = [
        "# Testes de Precisao do ST-GCN",
        "",
        f"- Modelo neural: `stgcn_poisson_nll`",
        f"- Janela temporal: `{payload['stgcn']['seq_len']}` semanas",
        f"- Ensemble: `{payload['stgcn']['ensemble_size']}` sementes",
        f"- Dimensao de entrada: `{payload['stgcn']['input_dim']}` canais",
        "",
        "| Motor | MSE | MAE |",
        "| :--- | ---: | ---: |",
        f"| poisson | {payload['poisson']['mse']:.6f} | {payload['poisson']['mae']:.6f} |",
        f"| stgcn_poisson_nll | {payload['stgcn']['mse']:.6f} | {payload['stgcn']['mae']:.6f} |",
        "",
        f"- Ganho absoluto de MSE: `{payload['comparison']['mse_gain']:.6f}`",
        f"- Ganho percentual de MSE: `{payload['comparison']['mse_gain_pct']:.2f}%`",
        f"- Ganho absoluto de MAE: `{payload['comparison']['mae_gain']:.6f}`",
        f"- Ganho percentual de MAE: `{payload['comparison']['mae_gain_pct']:.2f}%`",
    ]

    with open(STGCN_TESTS_MD_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def save_stgcn_dashboard_artifacts(result_payload):
    os.makedirs(os.path.dirname(STGCN_FORECASTS_CSV_PATH), exist_ok=True)
    forecasts_df = result_payload["forecasts_df"].copy()
    if not forecasts_df.empty:
        forecasts_df.to_csv(STGCN_FORECASTS_CSV_PATH, index=False)

    metadata = {
        key: value
        for key, value in result_payload.items()
        if key != "forecasts_df"
    }
    with open(STGCN_METADATA_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=str)


def load_stgcn_dashboard_artifacts():
    if not os.path.exists(STGCN_FORECASTS_CSV_PATH) or not os.path.exists(STGCN_METADATA_JSON_PATH):
        return None

    forecasts_df = pd.read_csv(STGCN_FORECASTS_CSV_PATH)
    with open(STGCN_METADATA_JSON_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    metadata["forecasts_df"] = forecasts_df
    return metadata
