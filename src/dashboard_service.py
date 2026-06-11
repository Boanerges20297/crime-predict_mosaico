import json
import math
import os
import re
import time
import unicodedata
from html import escape
from dataclasses import dataclass

import folium
import pandas as pd
from branca.colormap import LinearColormap
from folium import FeatureGroup, LayerControl
from folium.plugins import FastMarkerCluster
from shapely.geometry import shape

from benchmark_models import train_model_and_forecast
from genetic_algorithm import GeneticAlgorithmHex
from grid_generator import HexagonalGrid
from predictor import evaluate_model, prepare_weekly_series
from spatial_utils import (
    build_study_area,
    compute_bbox,
    filter_cvli,
    filter_fortaleza_bbox,
    iter_polygon_parts,
    load_processed_crimes,
)


DEFAULT_POP_SIZE = 6
DEFAULT_GENERATIONS = 5
DEFAULT_MUTATION_RATE = 0.15
DEFAULT_CROSSOVER_RATE = 0.8
DEFAULT_SEED = 42
DEFAULT_TARGET_HEX_COUNT = 180
DEFAULT_HEX_PENALTY_WEIGHT = 5.5
SPARSE_ACTIVITY_RATIO_FACTOR = 0.35
SPARSE_MIN_ACTIVITY_RATIO_FLOOR = 0.05
SPARSE_MIN_ACTIVE_WEEKS = 12
WEEKS_PER_TWO_MONTHS = (365.25 / 12 * 2) / 7
SPARSE_MIN_CVLI_PER_TWO_MONTHS = 1.0
KM_PER_LAT_DEGREE = 111.32

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEST_GRID_PATH = os.path.join(BASE_DIR, "data", "processed", "best_hex_grid.json")
BENCHMARK_JSON_PATH = os.path.join(BASE_DIR, "data", "processed", "benchmark_cvli_hex_models.json")
BAIRRO_POLYGONS_PATH = os.path.join(BASE_DIR, "data", "static", "bairros_fortaleza.geojson")
_CACHE = {}
_ANALYSIS_CACHE = {}
_BAIRRO_POLYGONS = None
_BAIRRO_NAME_ALIASES = {
    "BOA VISTA": "BOA VISTA / CASTELAO",
    "CASTELAO": "BOA VISTA / CASTELAO",
    "DENDE": "DENDE",
    "GENTILANDIA": "BENFICA",
    "JANGURURSSU": "JANGURUSSU",
    "PATRIOLINO RIBEIRO": "GUARARAPES",
    "PRAIA DO FURUTO II": "PRAIA DO FUTURO II",
    "PRAIA DE IRACEMA": "PRAIA DE IRACEMA",
    "SAPIRANGA COITE": "SAPIRANGA / COITE",
    "SAO JOAO DO TAUAPE": "TAUAPE",
    "VILA ELLERY": "ELLERY",
    "VILA PERY": "VILA PERI",
}


@dataclass
class AnalysisResult:
    filters: dict
    summary: dict
    metrics: dict
    best_config: dict | None
    map_html: str | None
    error: str | None = None
    from_cache: bool = False


def _normalize_region_name(value):
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.upper().strip()
    return re.sub(r"\s+", " ", text)


def _normalize_bairro_layer_name(value):
    normalized = _normalize_region_name(value)
    aliased = _BAIRRO_NAME_ALIASES.get(normalized, normalized)
    return _normalize_region_name(aliased)


def load_bairro_polygons():
    global _BAIRRO_POLYGONS
    if _BAIRRO_POLYGONS is not None:
        return _BAIRRO_POLYGONS

    with open(BAIRRO_POLYGONS_PATH, "r", encoding="utf-8") as file:
        payload = json.load(file)

    polygons = {}
    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        geometry = feature.get("geometry")
        if not geometry or geometry.get("type") not in {"Polygon", "MultiPolygon", "Polygon"}:
            continue

        display_name = str(properties.get("Name", "")).split(" - AIS ")[0].strip()
        normalized_name = _normalize_bairro_layer_name(display_name)
        if not normalized_name:
            continue

        polygons[normalized_name] = {
            "name": display_name,
            "geometry": shape(geometry),
        }

    _BAIRRO_POLYGONS = polygons
    return _BAIRRO_POLYGONS


def load_base_dataframe():
    df = load_processed_crimes()
    df = filter_cvli(df)
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = filter_fortaleza_bbox(df)
    return df.dropna(subset=["data", "latitude", "longitude"]).copy()


def load_saved_best_config():
    if not os.path.exists(BEST_GRID_PATH):
        return None
    with open(BEST_GRID_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def load_best_model_name():
    if not os.path.exists(BENCHMARK_JSON_PATH):
        return "poisson"
    with open(BENCHMARK_JSON_PATH, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload.get("best_model", "poisson")


def get_available_bairros(df):
    return sorted(df["bairro"].dropna().astype(str).unique().tolist())


def _normalize_bairros(selected_bairros):
    return tuple(sorted({bairro for bairro in selected_bairros if bairro}))


def filter_dataframe(df, bairros=None, start_date=None, end_date=None):
    filtered = df.copy()

    if bairros:
        filtered = filtered[filtered["bairro"].isin(bairros)].copy()

    if start_date:
        filtered = filtered[filtered["data"] >= pd.to_datetime(start_date)].copy()

    if end_date:
        filtered = filtered[filtered["data"] <= pd.to_datetime(end_date)].copy()

    return filtered


def build_cache_key(selected_bairros, start_date, end_date, pop_size, generations, hide_sparse_hexes, show_cvli_points, show_bairro_heatmap):
    return (
        _normalize_bairros(selected_bairros),
        str(start_date or ""),
        str(end_date or ""),
        int(pop_size),
        int(generations),
        bool(hide_sparse_hexes),
        bool(show_cvli_points),
        bool(show_bairro_heatmap),
    )


def build_analysis_cache_key(selected_bairros, start_date, end_date, pop_size, generations):
    return (
        _normalize_bairros(selected_bairros),
        str(start_date or ""),
        str(end_date or ""),
        int(pop_size),
        int(generations),
    )


def summarize_hex_activity(df_hex):
    if df_hex.empty:
        return pd.DataFrame(
            columns=[
                "hex_id",
                "total_cvli",
                "active_weeks",
                "total_weeks",
                "activity_ratio",
                "avg_cvli_per_two_months",
                "is_sparse",
            ]
        )

    temp = df_hex.copy()
    temp["data"] = pd.to_datetime(temp["data"])
    temp["semana"] = pd.to_datetime(temp["data"]).dt.to_period("W").dt.start_time
    weekly = temp.groupby(["hex_id", "semana"]).size().reset_index(name="cvli")
    total_weeks = int(weekly["semana"].nunique()) if not weekly.empty else 0

    summary = weekly.groupby("hex_id").agg(
        total_cvli=("cvli", "sum"),
        active_weeks=("semana", "nunique"),
    ).reset_index()
    summary["total_weeks"] = total_weeks
    summary["activity_ratio"] = summary["active_weeks"] / summary["total_weeks"].replace(0, 1)
    summary["avg_cvli_per_two_months"] = summary["total_cvli"] / (
        summary["total_weeks"].replace(0, 1) / WEEKS_PER_TWO_MONTHS
    )

    median_ratio = summary["activity_ratio"].median() if not summary.empty else 0
    sparse_ratio_threshold = max(median_ratio * SPARSE_ACTIVITY_RATIO_FACTOR, SPARSE_MIN_ACTIVITY_RATIO_FLOOR)
    summary["is_sparse"] = (
        (summary["activity_ratio"] < sparse_ratio_threshold)
        | (summary["active_weeks"] < SPARSE_MIN_ACTIVE_WEEKS)
        | (summary["avg_cvli_per_two_months"] < SPARSE_MIN_CVLI_PER_TWO_MONTHS)
    )
    return summary


def build_metrics(df_filtered, df_hex, best_config, grid, hex_summary, hide_sparse_hexes, forecast_df, best_model_name):
    sparse_hidden = int(hex_summary["is_sparse"].sum()) if not hex_summary.empty else 0
    metrics = {
        "records": int(len(df_filtered)),
        "bairros_ativos": int(df_filtered["bairro"].nunique()) if "bairro" in df_filtered else 0,
        "ais_ativas": int(df_filtered["ais"].nunique()) if "ais" in df_filtered else 0,
        "hexagonos_ativos": int(df_hex["hex_id"].nunique()) if len(df_hex) else 0,
        "hexagonos_criados": int(len(grid.hexagons)) if grid is not None else 0,
        "hexagonos_renderizados": int(len(grid.display_hexagons)) if grid is not None else 0,
        "hexagonos_esparsos": sparse_hidden,
        "hexagonos_visiveis_mapa": int(len(hex_summary[~hex_summary["is_sparse"]])) if hide_sparse_hexes and not hex_summary.empty else int(len(hex_summary)),
        "data_inicial": df_filtered["data"].min().date().isoformat() if len(df_filtered) else None,
        "data_final": df_filtered["data"].max().date().isoformat() if len(df_filtered) else None,
        "mse_hex": None,
        "mse_bairros": None,
        "mse_ais": None,
        "forecast_model": best_model_name,
    }

    if len(df_hex) > 10:
        df_series_hex = prepare_weekly_series(df_hex, region_col="hex_id", lags=3)
        metrics["mse_hex"] = evaluate_model(df_series_hex, region_col="hex_id", lags=3)

    if len(df_filtered) > 10:
        df_series_bairros = prepare_weekly_series(df_filtered, region_col="bairro", lags=3)
        metrics["mse_bairros"] = evaluate_model(df_series_bairros, region_col="bairro", lags=3)

        df_series_ais = prepare_weekly_series(df_filtered, region_col="ais", lags=3)
        metrics["mse_ais"] = evaluate_model(df_series_ais, region_col="ais", lags=3)

    if best_config:
        metrics["best_mse"] = best_config.get("best_mse")

    return metrics


def _build_cvli_pin_html():
    return """
    <div style="
        position: relative;
        width: 18px;
        height: 18px;
        transform: rotate(-45deg);
        border-radius: 50% 50% 50% 0;
        background: #2563eb;
        border: 2px solid #ffffff;
        box-shadow: 0 1px 6px rgba(15, 23, 42, 0.35);
    ">
        <div style="
            position: absolute;
            top: 50%;
            left: 50%;
            width: 7px;
            height: 7px;
            margin-left: -3.5px;
            margin-top: -3.5px;
            border-radius: 50%;
            background: #ef4444;
            transform: rotate(45deg);
        "></div>
    </div>
    """


def _build_map_legend_html(show_cvli_points=False, show_bairro_heatmap=False):
    cluster_section = ""
    if show_cvli_points:
        cluster_section = """
        <div style="margin-top:10px; padding-top:10px; border-top:1px solid #e2e8f0;">
        <div style="font-weight:700; margin-bottom:6px;">Clusters CVLI</div>
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                <span style="width:16px; height:16px; border-radius:50%; background:#c084fc; display:inline-block; box-shadow:0 0 0 4px rgba(236,72,153,0.18);"></span>
                <span>Baixa densidade no filtro</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                <span style="width:18px; height:18px; border-radius:50%; background:#a855f7; display:inline-block; box-shadow:0 0 0 5px rgba(236,72,153,0.24);"></span>
                <span>Densidade intermediaria</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="width:20px; height:20px; border-radius:50%; background:#7e22ce; display:inline-block; box-shadow:0 0 0 6px rgba(239,68,68,0.30);"></span>
                <span>Alta densidade no filtro</span>
            </div>
        </div>
        """

    bairro_section = ""
    if show_bairro_heatmap:
        bairro_section = """
        <div style="margin-top:10px; padding-top:10px; border-top:1px solid #e2e8f0;">
            <div style="font-weight:700; margin-bottom:6px;">Intensidade por bairro</div>
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                <span style="width:16px; height:16px; border-radius:4px; background:#f7f3df; border:1px solid #e6debb; display:inline-block;"></span>
                <span>Zero CVLI no recorte</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                <span style="width:16px; height:16px; border-radius:4px; background:#f3e7a1; border:1px solid #d6c979; display:inline-block;"></span>
                <span>Baixo histórico de CVLI</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                <span style="width:16px; height:16px; border-radius:4px; background:#e8c97a; border:1px solid #d0ae55; display:inline-block;"></span>
                <span>Médio histórico de CVLI</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                <span style="width:16px; height:16px; border-radius:4px; background:#d99a5e; border:1px solid #c27f43; display:inline-block;"></span>
                <span>Alto histórico de CVLI</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="width:16px; height:16px; border-radius:4px; background:#b42318; border:1px solid #7f1d1d; display:inline-block;"></span>
                <span>Altíssimo histórico de CVLI</span>
            </div>
        </div>
        """

    return f"""
    <div style="
        position: fixed;
        bottom: 26px;
        left: 26px;
        z-index: 9999;
        background: rgba(255,255,255,0.96);
        border: 1px solid #dbe3ef;
        border-radius: 14px;
        box-shadow: 0 10px 28px rgba(15,23,42,0.12);
        padding: 12px 14px;
        min-width: 220px;
        font-family: Inter, Arial, sans-serif;
        font-size: 12px;
        color: #0f172a;
        line-height: 1.35;
    ">
        <div style="font-weight:800; margin-bottom:8px;">Legenda</div>
        <div style="font-weight:700; margin-bottom:6px;">Intensidade dos hexagonos</div>
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
            <span style="width:16px; height:16px; border-radius:4px; background:#ddd6fe; border:1px solid #7e22ce; display:inline-block;"></span>
            <span>Menor concentracao de CVLI</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
            <span style="width:16px; height:16px; border-radius:4px; background:#a855f7; border:1px solid #7e22ce; display:inline-block;"></span>
            <span>Concentracao intermediaria</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <span style="width:16px; height:16px; border-radius:4px; background:#ef4444; border:1px solid #7e22ce; display:inline-block;"></span>
            <span>Maior concentracao de CVLI</span>
        </div>
        {bairro_section}
        {cluster_section}
    </div>
    """


def _build_popup_style_html():
    return """
    <style>
        .leaflet-popup-content-wrapper {
            border-radius: 14px;
        }
        .leaflet-popup-content {
            margin: 12px 14px;
        }
        .cvli-popup {
            min-width: 280px;
            max-width: 340px;
            font-family: Inter, Arial, sans-serif;
            font-size: 13px;
            line-height: 1.45;
            color: #1f2937;
        }
        .cvli-popup-title {
            font-size: 16px;
            font-weight: 800;
            margin-bottom: 8px;
            color: #111827;
        }
        .cvli-popup-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px 12px;
            margin-bottom: 10px;
        }
        .cvli-popup-item {
            display: flex;
            flex-direction: column;
        }
        .cvli-popup-label {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #6b7280;
        }
        .cvli-popup-value {
            font-size: 13px;
            font-weight: 600;
            color: #111827;
        }
        .cvli-popup-section {
            border-top: 1px solid #e5e7eb;
            padding-top: 8px;
            margin-top: 8px;
        }
        .cvli-popup-meta {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px 12px;
        }
    </style>
    """


def estimate_hex_dimensions_km(radius_degrees, latitude):
    km_per_lon_degree = KM_PER_LAT_DEGREE * math.cos(math.radians(latitude))
    radius_km = radius_degrees * (KM_PER_LAT_DEGREE + km_per_lon_degree) / 2
    width_km = 2 * radius_km
    height_km = math.sqrt(3) * radius_km
    area_km2 = (3 * math.sqrt(3) / 2) * (radius_km ** 2)
    return {
        "radius_km": radius_km,
        "width_km": width_km,
        "height_km": height_km,
        "area_km2": area_km2,
    }


def build_map(
    df_records,
    df_hex,
    grid,
    study_area,
    hide_sparse_hexes=False,
    hex_summary=None,
    forecast_df=None,
    show_cvli_points=False,
    show_bairro_heatmap=False,
):
    lat_center = df_records["latitude"].mean()
    lon_center = df_records["longitude"].mean()
    map_object = folium.Map(location=[lat_center, lon_center], zoom_start=12, tiles="cartodbpositron")

    group_hex = FeatureGroup(name="Hexágonos ótimos", show=True)
    group_bairros = FeatureGroup(name="Centroides de bairros", show=False)
    group_bairro_heatmap = FeatureGroup(name="Histórico de CVLI por bairro", show=show_bairro_heatmap)

    hex_counts = df_hex["hex_id"].value_counts().to_dict()
    max_count = max(hex_counts.values()) if hex_counts else 1
    color_scale = LinearColormap(
        colors=["#ddd6fe", "#a855f7", "#ec4899", "#ef4444"],
        vmin=0,
        vmax=max_count,
    )

    sparse_lookup = {}
    if hex_summary is not None and not hex_summary.empty:
        sparse_lookup = hex_summary.set_index("hex_id")["is_sparse"].to_dict()
        summary_lookup = hex_summary.set_index("hex_id").to_dict("index")
    else:
        summary_lookup = {}

    forecast_lookup = {}
    if forecast_df is not None and not forecast_df.empty:
        forecast_lookup = forecast_df.set_index("hex_id").to_dict("index")

    visible_hex_ids = {hex_id for hex_id in hex_counts if hex_id != -1}
    if hide_sparse_hexes and sparse_lookup:
        visible_hex_ids = {hex_id for hex_id in visible_hex_ids if not sparse_lookup.get(hex_id, False)}

    hex_dimensions = estimate_hex_dimensions_km(grid.R, lat_center)

    for hex_id, count in hex_counts.items():
        if hex_id == -1:
            continue
        if hide_sparse_hexes and sparse_lookup.get(hex_id, False):
            continue

        geometry = grid.display_hexagons[hex_id]
        normalized = count / max(max_count, 1)
        fill_color = color_scale(count)
        fill_opacity = 0.05 + (0.11 * normalized)
        hex_meta = summary_lookup.get(hex_id, {})
        hex_forecast = forecast_lookup.get(hex_id, {})
        total_cvli = int(hex_meta.get("total_cvli", count))
        active_weeks = int(hex_meta.get("active_weeks", 0))
        total_weeks = int(hex_meta.get("total_weeks", 0))
        activity_ratio = float(hex_meta.get("activity_ratio", 0.0))
        avg_cvli_week = total_cvli / max(active_weeks, 1)
        forecast_value = float(hex_forecast.get("previsao_proxima_semana", 0.0))
        last_count = int(hex_forecast.get("ultima_contagem", 0))
        last_week = str(hex_forecast.get("ultima_semana_observada", "-"))
        moving_avg_4 = float(hex_forecast.get("media_movel_4", 0.0))
        trend_1 = float(hex_forecast.get("tendencia_1", 0.0))
        model_name = str(hex_forecast.get("modelo_previsao", "poisson"))
        trend_label = "alta" if trend_1 > 0 else "queda" if trend_1 < 0 else "estável"
        popup_html = (
            "<div class='cvli-popup'>"
            f"<div class='cvli-popup-title'>Hexágono {hex_id}</div>"
            "<div class='cvli-popup-grid'>"
            f"<div class='cvli-popup-item'><span class='cvli-popup-label'>CVLI total</span><span class='cvli-popup-value'>{total_cvli}</span></div>"
            f"<div class='cvli-popup-item'><span class='cvli-popup-label'>Atividade</span><span class='cvli-popup-value'>{activity_ratio:.2%}</span></div>"
            f"<div class='cvli-popup-item'><span class='cvli-popup-label'>Semanas ativas</span><span class='cvli-popup-value'>{active_weeks} / {total_weeks}</span></div>"
            f"<div class='cvli-popup-item'><span class='cvli-popup-label'>Média semanal ativa</span><span class='cvli-popup-value'>{avg_cvli_week:.2f}</span></div>"
            f"<div class='cvli-popup-item'><span class='cvli-popup-label'>Última contagem</span><span class='cvli-popup-value'>{last_count}</span></div>"
            f"<div class='cvli-popup-item'><span class='cvli-popup-label'>Próxima semana</span><span class='cvli-popup-value'>{forecast_value:.3f}</span></div>"
            "</div>"
            "<div class='cvli-popup-section'>"
            "<div class='cvli-popup-meta'>"
            f"<div class='cvli-popup-item'><span class='cvli-popup-label'>Última semana</span><span class='cvli-popup-value'>{last_week}</span></div>"
            f"<div class='cvli-popup-item'><span class='cvli-popup-label'>Tendência</span><span class='cvli-popup-value'>{trend_label} ({trend_1:+.2f})</span></div>"
            f"<div class='cvli-popup-item'><span class='cvli-popup-label'>MM4</span><span class='cvli-popup-value'>{moving_avg_4:.2f}</span></div>"
            f"<div class='cvli-popup-item'><span class='cvli-popup-label'>Modelo</span><span class='cvli-popup-value'>{model_name}</span></div>"
            f"<div class='cvli-popup-item'><span class='cvli-popup-label'>Esparso</span><span class='cvli-popup-value'>{'Sim' if sparse_lookup.get(hex_id, False) else 'Não'}</span></div>"
            f"<div class='cvli-popup-item'><span class='cvli-popup-label'>Área aprox.</span><span class='cvli-popup-value'>{hex_dimensions['area_km2']:.2f} km²</span></div>"
            f"<div class='cvli-popup-item'><span class='cvli-popup-label'>Largura</span><span class='cvli-popup-value'>{hex_dimensions['width_km']:.2f} km</span></div>"
            f"<div class='cvli-popup-item'><span class='cvli-popup-label'>Altura</span><span class='cvli-popup-value'>{hex_dimensions['height_km']:.2f} km</span></div>"
            "</div>"
            "</div>"
            "</div>"
        )

        for polygon in iter_polygon_parts(geometry):
            coords = [[lat, lon] for lon, lat in polygon.exterior.coords]
            folium.Polygon(
                locations=coords,
                color="#7E22CE",
                weight=0.95,
                fill=True,
                fill_color=fill_color,
                fill_opacity=fill_opacity,
                popup=folium.Popup(popup_html, max_width=360),
            ).add_to(group_hex)

    if show_cvli_points:
        df_points = df_hex.copy()
        if hide_sparse_hexes:
            df_points = df_points[df_points["hex_id"].isin(visible_hex_ids)].copy()
        if not df_points.empty:
            max_point_date = pd.to_datetime(df_points["data"]).max()
            min_point_date = max_point_date - pd.DateOffset(years=1)
            df_points = df_points[pd.to_datetime(df_points["data"]) >= min_point_date].copy()

        max_cluster_count = int(df_points["hex_id"].value_counts().max()) if len(df_points) else 1
        max_cluster_count = max(max_cluster_count, 1)
        cluster_data = []
        for record in df_points.itertuples():
            popup_parts = ["<b>CVLI</b>"]
            if hasattr(record, "bairro") and pd.notna(record.bairro):
                popup_parts.append(f"Bairro: {escape(str(record.bairro))}")
            if hasattr(record, "ais") and pd.notna(record.ais):
                popup_parts.append(f"AIS: {escape(str(record.ais))}")
            if hasattr(record, "data") and pd.notna(record.data):
                popup_parts.append(f"Data: {pd.to_datetime(record.data).date().isoformat()}")
            cluster_data.append([float(record.latitude), float(record.longitude), "<br>".join(popup_parts)])

        marker_callback = """
            function(row) {
                var pointIcon = L.divIcon({
                    html: '<div style="'
                        + 'position:relative;'
                        + 'width:14px;'
                        + 'height:14px;'
                        + 'transform:rotate(-45deg);'
                        + 'border-radius:50% 50% 50% 0;'
                        + 'background:#2563eb;'
                        + 'border:2px solid #ffffff;'
                        + 'box-shadow:0 1px 5px rgba(15,23,42,0.30);'
                        + '"><div style="'
                        + 'position:absolute;'
                        + 'top:50%;'
                        + 'left:50%;'
                        + 'width:6px;'
                        + 'height:6px;'
                        + 'margin-left:-3px;'
                        + 'margin-top:-3px;'
                        + 'border-radius:50%;'
                        + 'background:#ef4444;'
                        + 'transform:rotate(45deg);'
                        + '"></div></div>',
                    className: 'cvli-point-icon',
                    iconSize: [14, 14],
                    iconAnchor: [7, 14]
                });
                return L.marker(new L.LatLng(row[0], row[1]), {icon: pointIcon}).bindPopup(row[2]);
            }
        """
        cluster_icon_function = f"""
            function(cluster) {{
                var count = cluster.getChildCount();
                var maxCount = {max_cluster_count};
                var ratio = Math.max(0, Math.min(1, count / maxCount));
                var size = Math.round(32 + (ratio * 20));
                var fontSize = Math.round(12 + (ratio * 3));
                var hue = 275 - (ratio * 80);
                var sat = 78 + (ratio * 10);
                var light = 72 - (ratio * 22);
                var background = 'hsl(' + hue + ', ' + sat + '%, ' + light + '%)';
                var ringAlpha = 0.18 + (ratio * 0.18);
                var ring = 'hsla(' + (335 - ratio * 20) + ', 84%, 60%, ' + ringAlpha.toFixed(2) + ')';
                return L.divIcon({{
                    html: '<div style="'
                        + 'width:' + size + 'px;'
                        + 'height:' + size + 'px;'
                        + 'border-radius:50%;'
                        + 'background:' + background + ';'
                        + 'border:2px solid #ffffff;'
                        + 'box-shadow:0 0 0 6px ' + ring + ', 0 2px 10px rgba(15,23,42,0.18);'
                        + 'display:flex;'
                        + 'align-items:center;'
                        + 'justify-content:center;'
                        + 'color:#ffffff;'
                        + 'font-weight:800;'
                        + 'font-size:' + fontSize + 'px;'
                        + 'font-family:Inter, Arial, sans-serif;'
                        + '">' + count + '</div>',
                    className: 'cvli-cluster-icon',
                    iconSize: [size, size]
                }});
            }}
        """
        FastMarkerCluster(
            data=cluster_data,
            callback=marker_callback,
            name="Clusters CVLI",
            show=True,
            icon_create_function=cluster_icon_function,
            options={
                "showCoverageOnHover": False,
                "spiderfyOnMaxZoom": True,
                "removeOutsideVisibleBounds": True,
                "animate": False,
                "chunkedLoading": True,
                "chunkInterval": 120,
                "chunkDelay": 40,
                "maxClusterRadius": 80,
                "disableClusteringAtZoom": 17,
            },
        ).add_to(map_object)

    if show_bairro_heatmap:
        bairro_polygons = load_bairro_polygons()
        bairro_counts = (
            df_records["bairro"]
            .fillna("")
            .astype(str)
            .map(_normalize_bairro_layer_name)
            .value_counts()
            .to_dict()
        )
        positive_counts = sorted(count for count in bairro_counts.values() if count > 0)
        if positive_counts:
            low_threshold = positive_counts[max(0, int(len(positive_counts) * 0.25) - 1)]
            mid_threshold = positive_counts[max(0, int(len(positive_counts) * 0.50) - 1)]
            high_threshold = positive_counts[max(0, int(len(positive_counts) * 0.75) - 1)]
        else:
            low_threshold = 0
            mid_threshold = 0
            high_threshold = 0

        def get_bairro_fill(count):
            if count <= 0:
                return "#f7f3df", 0.20
            if count <= low_threshold:
                return "#f3e7a1", 0.30
            if count <= mid_threshold:
                return "#e8c97a", 0.38
            if count <= high_threshold:
                return "#d99a5e", 0.47
            return "#b42318", 0.56

        for bairro_key, polygon_entry in bairro_polygons.items():
            count = int(bairro_counts.get(bairro_key, 0))
            fill_color, fill_opacity = get_bairro_fill(count)
            popup_content = (
                "<div class='cvli-popup'>"
                f"<b>{escape(polygon_entry['name'])}</b><br>"
                f"CVLI no filtro atual: {int(count)}"
                "</div>"
            )

            for polygon in iter_polygon_parts(polygon_entry["geometry"]):
                coords = [[coord[1], coord[0]] for coord in polygon.exterior.coords]
                folium.Polygon(
                    locations=coords,
                    color="#991B1B",
                    weight=1.1,
                    fill=True,
                    fill_color=fill_color,
                    fill_opacity=fill_opacity,
                    popup=folium.Popup(popup_content, max_width=320),
                    tooltip=f"{polygon_entry['name']}: {int(count)} CVLI",
                ).add_to(group_bairro_heatmap)

    bairro_centroids = df_records.groupby("bairro")[["latitude", "longitude"]].mean()
    bairro_counts = df_records["bairro"].value_counts()
    for bairro_name, row in bairro_centroids.iterrows():
        count = int(bairro_counts.get(bairro_name, 0))
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=max(4, min(12, 4 + count / 200)),
            color="#0F766E",
            fill=True,
            fill_color="#14B8A6",
            fill_opacity=0.6,
            popup=f"Bairro: {bairro_name}<br>Ocorrências: {count}",
        ).add_to(group_bairros)

    group_hex.add_to(map_object)
    group_bairros.add_to(map_object)
    if show_bairro_heatmap:
        group_bairro_heatmap.add_to(map_object)
    LayerControl(collapsed=False).add_to(map_object)
    map_object.get_root().header.add_child(folium.Element(_build_popup_style_html()))
    map_object.get_root().html.add_child(
        folium.Element(
            _build_map_legend_html(
                show_cvli_points=show_cvli_points,
                show_bairro_heatmap=show_bairro_heatmap,
            )
        )
    )
    return map_object._repr_html_()


def analyze_filters(
    df_base,
    selected_bairros=None,
    start_date=None,
    end_date=None,
    pop_size=DEFAULT_POP_SIZE,
    generations=DEFAULT_GENERATIONS,
    hide_sparse_hexes=False,
    show_cvli_points=False,
    show_bairro_heatmap=False,
):
    selected_bairros = list(selected_bairros or [])
    cache_key = build_cache_key(
        selected_bairros,
        start_date,
        end_date,
        pop_size,
        generations,
        hide_sparse_hexes,
        show_cvli_points,
        show_bairro_heatmap,
    )
    if cache_key in _CACHE:
        cached = _CACHE[cache_key]
        return AnalysisResult(**(cached | {"from_cache": True}))

    started_at = time.time()
    df_filtered = filter_dataframe(df_base, bairros=selected_bairros, start_date=start_date, end_date=end_date)

    filters = {
        "selected_bairros": selected_bairros,
        "start_date": start_date or "",
        "end_date": end_date or "",
        "pop_size": int(pop_size),
        "generations": int(generations),
        "hide_sparse_hexes": bool(hide_sparse_hexes),
        "show_cvli_points": bool(show_cvli_points),
        "show_bairro_heatmap": bool(show_bairro_heatmap),
    }

    if df_filtered.empty:
        return AnalysisResult(filters=filters, summary={}, metrics={}, best_config=None, map_html=None, error="Nenhum registro encontrado para os filtros selecionados.")

    if len(df_filtered) < 200:
        return AnalysisResult(
            filters=filters,
            summary={},
            metrics={"records": int(len(df_filtered))},
            best_config=None,
            map_html=None,
            error="Poucos registros para otimização confiável. Amplie o intervalo de datas ou selecione mais bairros.",
        )

    analysis_cache_key = build_analysis_cache_key(selected_bairros, start_date, end_date, pop_size, generations)
    if analysis_cache_key in _ANALYSIS_CACHE:
        analysis = _ANALYSIS_CACHE[analysis_cache_key]
        df_filtered = analysis["df_filtered"]
        df_hex = analysis["df_hex"]
        grid = analysis["grid"]
        study_area = analysis["study_area"]
        best_config = analysis["best_config"]
        hex_summary = analysis["hex_summary"]
        forecast_df = analysis["forecast_df"]
        best_model_name = analysis["best_model_name"]
        base_metrics = analysis["base_metrics"]
        summary = analysis["summary"]
        use_saved_default = analysis["use_saved_default"]
    else:
        bbox = compute_bbox(df_filtered)
        study_area = build_study_area(df_filtered)
        use_saved_default = not selected_bairros and not start_date and not end_date
        best_config = load_saved_best_config() if use_saved_default else None

        if best_config is None:
            ga = GeneticAlgorithmHex(
                df=df_filtered,
                bbox=bbox,
                study_area=study_area,
                pop_size=int(pop_size),
                generations=int(generations),
                mutation_rate=DEFAULT_MUTATION_RATE,
                crossover_rate=DEFAULT_CROSSOVER_RATE,
                seed=DEFAULT_SEED,
                target_hex_count=DEFAULT_TARGET_HEX_COUNT,
                hex_penalty_weight=DEFAULT_HEX_PENALTY_WEIGHT,
            )
            best_config = ga.run()

        grid = HexagonalGrid(
            bbox,
            dx=best_config["dx"],
            dy=best_config["dy"],
            theta=best_config["theta"],
            R=best_config["R"],
            study_area=study_area,
        )

        df_hex = df_filtered.copy()
        df_hex["hex_id"] = grid.assign_points(df_hex)
        df_hex = df_hex[df_hex["hex_id"] != -1].copy()
        hex_summary = summarize_hex_activity(df_hex)
        best_model_name = load_best_model_name()
        weekly_series = prepare_weekly_series(df_hex, region_col="hex_id", lags=3)
        _, forecast_df = train_model_and_forecast(weekly_series, best_model_name, region_col="hex_id", lags=3)

        base_metrics = build_metrics(df_filtered, df_hex, best_config, grid, hex_summary, False, forecast_df, best_model_name)
        base_metrics["assigned_records"] = int(len(df_hex))
        base_metrics["used_saved_config"] = use_saved_default and load_saved_best_config() is not None

        summary = {
            "bbox": tuple(float(value) for value in bbox),
            "selected_bairros_count": len(selected_bairros),
            "target_hex_count": DEFAULT_TARGET_HEX_COUNT,
        }
        _ANALYSIS_CACHE[analysis_cache_key] = {
            "df_filtered": df_filtered,
            "df_hex": df_hex,
            "grid": grid,
            "study_area": study_area,
            "best_config": best_config,
            "hex_summary": hex_summary,
            "forecast_df": forecast_df,
            "best_model_name": best_model_name,
            "base_metrics": base_metrics,
            "summary": summary,
            "use_saved_default": use_saved_default,
        }

    metrics = base_metrics.copy()
    metrics["hexagonos_visiveis_mapa"] = int(len(hex_summary[~hex_summary["is_sparse"]])) if hide_sparse_hexes and not hex_summary.empty else int(len(hex_summary))
    metrics["execution_seconds"] = round(time.time() - started_at, 2)

    map_html = build_map(
        df_filtered,
        df_hex,
        grid,
        study_area,
        hide_sparse_hexes=hide_sparse_hexes,
        hex_summary=hex_summary,
        forecast_df=forecast_df,
        show_cvli_points=show_cvli_points,
        show_bairro_heatmap=show_bairro_heatmap,
    )
    payload = {
        "filters": filters,
        "summary": summary,
        "metrics": metrics,
        "best_config": best_config,
        "map_html": map_html,
        "error": None,
        "from_cache": False,
    }
    _CACHE[cache_key] = payload
    return AnalysisResult(**payload)
