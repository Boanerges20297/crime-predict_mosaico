from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "figuras_artigo"

SUBDIRS = {
    "coverage": OUTPUT_DIR / "cobertura_95",
    "scatter": OUTPUT_DIR / "dispersao",
    "moving_average": OUTPUT_DIR / "media_movel",
    "median": OUTPUT_DIR / "mediana",
    "series": OUTPUT_DIR / "series_temporais",
    "hex": OUTPUT_DIR / "hexagonos",
}

STYLE = {
    "bg": "#f6f1e8",
    "grid": "#d9ceb8",
    "text": "#1f1f1f",
    "accent": "#c26d1a",
    "accent_2": "#245c73",
    "accent_3": "#8a3b12",
    "accent_4": "#3f7d20",
}


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for path in SUBDIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def apply_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": STYLE["bg"],
            "axes.facecolor": STYLE["bg"],
            "axes.edgecolor": STYLE["grid"],
            "axes.labelcolor": STYLE["text"],
            "axes.titlecolor": STYLE["text"],
            "axes.grid": True,
            "grid.color": STYLE["grid"],
            "grid.alpha": 0.6,
            "xtick.color": STYLE["text"],
            "ytick.color": STYLE["text"],
            "text.color": STYLE["text"],
            "font.size": 11,
        }
    )


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_coverage() -> None:
    coverage = pd.read_csv(DATA_DIR / "baseline_bairros_95_coverage.csv")
    selected = coverage.loc[coverage["selected"]].head(15).copy()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(selected["bairro"], selected["event_count"], color=STYLE["accent_2"])
    ax.invert_yaxis()
    ax.set_title("Top 15 bairros selecionados na cobertura acumulada de 95%")
    ax.set_xlabel("Quantidade de eventos")
    ax.set_ylabel("Bairro")

    for idx, (_, row) in enumerate(selected.iterrows()):
        ax.text(
            row["event_count"] + 1,
            idx,
            f"{row['cumulative_share'] * 100:.1f}%",
            va="center",
            fontsize=9,
        )

    save_figure(fig, SUBDIRS["coverage"] / "top15_bairros_cobertura_95.png")


def plot_predictions_scatter() -> None:
    preds = pd.read_csv(DATA_DIR / "baseline_bairros_95_predictions.csv")
    sample = preds.sample(min(len(preds), 3000), random_state=42).copy()

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        sample["y_true"],
        sample["y_pred"],
        s=26,
        alpha=0.28,
        color=STYLE["accent"],
        edgecolors="none",
    )
    max_value = max(float(sample["y_true"].max()), float(sample["y_pred"].max()))
    ax.plot([0, max_value], [0, max_value], linestyle="--", color=STYLE["accent_3"], linewidth=1.5)
    ax.set_title("Dispersao entre observado e previsto nos bairros da cobertura de 95%")
    ax.set_xlabel("CVLI observado por semana")
    ax.set_ylabel("CVLI previsto por semana")

    save_figure(fig, SUBDIRS["scatter"] / "predito_vs_observado_bairros_95.png")


def load_weekly_cvli() -> pd.DataFrame:
    crimes = pd.read_csv(DATA_DIR / "fortaleza_crimes_normalizado.csv", low_memory=False)
    crimes["data"] = pd.to_datetime(crimes["data"], errors="coerce")

    cvli = crimes.loc[
        (crimes["cidade"].fillna("").str.upper() == "FORTALEZA")
        & (crimes["tipo"].fillna("").str.lower() == "cvli")
        & (crimes["data"].notna())
    ].copy()

    weekly = (
        cvli.assign(semana=cvli["data"].dt.to_period("W").dt.start_time)
        .groupby("semana")
        .size()
        .rename("cvli_total")
        .reset_index()
        .sort_values("semana")
    )
    weekly["media_movel_4s"] = weekly["cvli_total"].rolling(window=4, min_periods=1).mean()
    weekly["mediana_movel_4s"] = weekly["cvli_total"].rolling(window=4, min_periods=1).median()
    return weekly


def plot_weekly_series(weekly: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(weekly["semana"], weekly["cvli_total"], color=STYLE["accent_2"], linewidth=1.5)
    ax.set_title("Serie semanal de CVLI em Fortaleza")
    ax.set_xlabel("Semana")
    ax.set_ylabel("Ocorrencias")
    save_figure(fig, SUBDIRS["series"] / "cvli_semanal_fortaleza.png")


def plot_moving_average(weekly: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(weekly["semana"], weekly["cvli_total"], color=STYLE["grid"], linewidth=1.2, label="Serie semanal")
    ax.plot(weekly["semana"], weekly["media_movel_4s"], color=STYLE["accent"], linewidth=2.5, label="Media movel 4 semanas")
    ax.set_title("CVLI semanal com media movel de 4 semanas")
    ax.set_xlabel("Semana")
    ax.set_ylabel("Ocorrencias")
    ax.legend(frameon=False)
    save_figure(fig, SUBDIRS["moving_average"] / "cvli_media_movel_4_semanas.png")


def plot_moving_median(weekly: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(weekly["semana"], weekly["cvli_total"], color=STYLE["grid"], linewidth=1.2, label="Serie semanal")
    ax.plot(weekly["semana"], weekly["mediana_movel_4s"], color=STYLE["accent_4"], linewidth=2.5, label="Mediana movel 4 semanas")
    ax.set_title("CVLI semanal com mediana movel de 4 semanas")
    ax.set_xlabel("Semana")
    ax.set_ylabel("Ocorrencias")
    ax.legend(frameon=False)
    save_figure(fig, SUBDIRS["median"] / "cvli_mediana_movel_4_semanas.png")


def plot_hex_sweep() -> None:
    sweep = json.loads((DATA_DIR / "cvli_hex_sweep.json").read_text(encoding="utf-8"))
    targets = pd.DataFrame(sweep["targets"]).sort_values("target_hex_count")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(targets["target_hex_count"], targets["best_quick_mse"], marker="o", color=STYLE["accent_3"], label="MSE")
    ax.plot(targets["target_hex_count"], targets["best_quick_mae"], marker="s", color=STYLE["accent_2"], label="MAE")
    ax.set_title("Erro dos modelos rapidos por meta de hexagonos")
    ax.set_xlabel("Meta de hexagonos")
    ax.set_ylabel("Erro")
    ax.legend(frameon=False)
    save_figure(fig, SUBDIRS["hex"] / "sweep_hexagonos_mse_mae.png")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        targets["target_hex_count"],
        targets["mean_activity_ratio"],
        marker="o",
        color=STYLE["accent"],
        label="Media de atividade",
    )
    ax.plot(
        targets["target_hex_count"],
        targets["median_activity_ratio"],
        marker="s",
        color=STYLE["accent_4"],
        label="Mediana de atividade",
    )
    ax.set_title("Estabilidade de atividade por configuracao de hexagonos")
    ax.set_xlabel("Meta de hexagonos")
    ax.set_ylabel("Razao de semanas com CVLI")
    ax.legend(frameon=False)
    save_figure(fig, SUBDIRS["hex"] / "sweep_hexagonos_atividade.png")


def write_readme() -> None:
    lines = [
        "# Figuras do artigo",
        "",
        "Esta pasta concentra figuras exportadas a partir dos artefatos do projeto para apoiar a redacao do artigo.",
        "",
        "## Subpastas",
        "- `cobertura_95`: graficos sobre selecao dos bairros que compoem 95% da massa de eventos.",
        "- `dispersao`: comparacoes entre valores observados e previstos.",
        "- `media_movel`: series suavizadas por media movel.",
        "- `mediana`: series suavizadas por mediana movel.",
        "- `series_temporais`: curvas brutas agregadas no tempo.",
        "- `hexagonos`: diagnosticos do sweep de configuracoes espaciais.",
        "",
        "## Como regenerar",
        "Execute:",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe .\\src\\generate_article_figures.py",
        "```",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    apply_style()
    plot_coverage()
    plot_predictions_scatter()
    weekly = load_weekly_cvli()
    plot_weekly_series(weekly)
    plot_moving_average(weekly)
    plot_moving_median(weekly)
    plot_hex_sweep()
    write_readme()
    print(f"Figuras geradas em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
