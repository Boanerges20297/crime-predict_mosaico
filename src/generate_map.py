import json

import folium
from folium import FeatureGroup, LayerControl

from benchmark_models import train_model_and_forecast
from grid_generator import HexagonalGrid
from predictor import prepare_weekly_series
from spatial_utils import (
    build_study_area,
    compute_bbox,
    filter_cvli,
    filter_fortaleza_bbox,
    get_processed_data_path,
    iter_polygon_parts,
    load_processed_crimes,
)


def main():
    data_path = get_processed_data_path()
    print(f"Carregando base de crimes para o mapa: {data_path}")
    df = load_processed_crimes()
    df = filter_cvli(df)
    df_clean = filter_fortaleza_bbox(df)

    with open("data/processed/best_hex_grid.json", "r", encoding="utf-8") as file:
        best_hex = json.load(file)

    bbox = compute_bbox(df_clean)
    study_area = build_study_area(df_clean)

    grid = HexagonalGrid(
        bbox,
        dx=best_hex["dx"],
        dy=best_hex["dy"],
        theta=best_hex["theta"],
        R=best_hex["R"],
        study_area=study_area,
    )
    df_clean["hex_id"] = grid.assign_points(df_clean)
    df_hex = df_clean[df_clean["hex_id"] != -1].copy()

    weekly_series = prepare_weekly_series(df_hex, region_col="hex_id", lags=3)
    _, forecast_df = train_model_and_forecast(weekly_series, "poisson", region_col="hex_id", lags=3)
    forecast_lookup = forecast_df.set_index("hex_id").to_dict("index") if not forecast_df.empty else {}

    lat_center = df_clean["latitude"].mean()
    lon_center = df_clean["longitude"].mean()
    map_object = folium.Map(location=[lat_center, lon_center], zoom_start=12, tiles="cartodbpositron")

    group_ais = FeatureGroup(name="1. Áreas Integradas de Segurança (AIS)", show=True)
    group_bairros = FeatureGroup(name="2. Divisão por Bairros", show=False)
    group_hex = FeatureGroup(name="3. Mosaico Hexagonal Ótimo (AG)", show=False)

    print("Mapeando agrupamentos de AIS...")
    ais_centroids = df_clean.groupby("ais")[["latitude", "longitude"]].mean()
    ais_counts = df_clean["ais"].value_counts()
    for ais_name, row in ais_centroids.iterrows():
        count = ais_counts.get(ais_name, 0)
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=15 + (count / 1000),
            color="#FF4B4B",
            fill=True,
            fill_color="#FF4B4B",
            fill_opacity=0.6,
            popup=f"Região: {ais_name}<br>Total Ocorrências: {count}",
        ).add_to(group_ais)

    print("Mapeando agrupamentos de Bairros...")
    bairro_centroids = df_clean.groupby("bairro")[["latitude", "longitude"]].mean()
    bairro_counts = df_clean["bairro"].value_counts()
    for bairro_name, row in bairro_centroids.iterrows():
        count = bairro_counts.get(bairro_name, 0)
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6 + (count / 300),
            color="#3B82F6",
            fill=True,
            fill_color="#3B82F6",
            fill_opacity=0.6,
            popup=f"Bairro: {bairro_name}<br>Total Ocorrências: {count}",
        ).add_to(group_bairros)

    print("Mapeando polígonos do Mosaico Hexagonal Ótimo...")
    hex_counts = df_hex["hex_id"].value_counts().to_dict()
    max_count = max(hex_counts.values()) if hex_counts else 1

    for hex_id, count in hex_counts.items():
        geometry = grid.display_hexagons[hex_id]
        color_intensity = min(int(50 + (count / max_count) * 205), 255)
        hex_color = f"#{color_intensity:02x}10a0"

        forecast_meta = forecast_lookup.get(hex_id, {})
        forecast_value = float(forecast_meta.get("previsao_proxima_semana", 0.0))
        last_count = int(forecast_meta.get("ultima_contagem", 0))
        last_week = str(forecast_meta.get("ultima_semana_observada", "-"))
        moving_avg_4 = float(forecast_meta.get("media_movel_4", 0.0))
        trend_1 = float(forecast_meta.get("tendencia_1", 0.0))
        model_name = str(forecast_meta.get("modelo_previsao", "poisson"))
        trend_label = "alta" if trend_1 > 0 else "queda" if trend_1 < 0 else "estável"
        popup_html = (
            f"<b>Hexágono {hex_id}</b><br>"
            f"CVLI total no período: {count}<br>"
            f"Última semana observada: {last_week}<br>"
            f"Última contagem semanal: {last_count}<br>"
            f"MM4 recente: {moving_avg_4:.2f}<br>"
            f"Tendência recente: {trend_label} ({trend_1:+.2f})<br>"
            f"Previsão próxima semana: {forecast_value:.3f}<br>"
            f"Modelo preditivo: {model_name}<br>"
            f"Hexágonos criados na malha: {len(grid.hexagons)}"
        )

        for polygon in iter_polygon_parts(geometry):
            coords = [[lat, lon] for lon, lat in polygon.exterior.coords]
            folium.Polygon(
                locations=coords,
                color="#A855F7",
                weight=1,
                fill=True,
                fill_color=hex_color,
                fill_opacity=0.24,
                popup=popup_html,
            ).add_to(group_hex)

    group_ais.add_to(map_object)
    group_bairros.add_to(map_object)
    group_hex.add_to(map_object)
    LayerControl(collapsed=False).add_to(map_object)

    output_html = "mapa_interativo_fortaleza.html"
    map_object.save(output_html)
    print(f"Mapa interativo gerado com sucesso em '{output_html}'")


if __name__ == "__main__":
    main()
