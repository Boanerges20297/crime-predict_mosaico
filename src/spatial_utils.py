import os

import numpy as np
import pandas as pd
from shapely.geometry import GeometryCollection, MultiPoint, MultiPolygon, Point, Polygon
from sklearn.cluster import DBSCAN

try:
    from shapely import concave_hull as shapely_concave_hull
except ImportError:
    shapely_concave_hull = None


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORTALEZA_LAT_RANGE = (-3.9, -3.6)
FORTALEZA_LON_RANGE = (-38.7, -38.4)
PROCESSED_DATA_CANDIDATES = [
    os.path.join(BASE_DIR, "data", "processed", "fortaleza_crimes_normalizado.csv"),
    os.path.join(BASE_DIR, "data", "processed", "fortaleza_crimes.csv"),
]


def get_processed_data_path():
    for path in PROCESSED_DATA_CANDIDATES:
        if os.path.exists(path):
            return path

    searched = "\n - ".join(PROCESSED_DATA_CANDIDATES)
    raise FileNotFoundError(f"Base processada não encontrada. Caminhos verificados:\n - {searched}")


def load_processed_crimes():
    return pd.read_csv(get_processed_data_path(), low_memory=False)


def filter_cvli(df):
    if "tipo" in df.columns:
        mask = df["tipo"].fillna("").astype(str).str.upper().eq("CVLI")
        return df[mask].copy()

    event_cols = [col for col in ["tipo_evento", "nature"] if col in df.columns]
    if not event_cols:
        return df.copy()

    patterns = ["HOMICIDIO DOLOSO", "FEMINICIDIO", "LATROCINIO", "LESAO CORPORAL SEGUIDA DE MORTE"]
    mask = pd.Series(False, index=df.index)
    for col in event_cols:
        upper = df[col].fillna("").astype(str).str.upper()
        for pattern in patterns:
            mask = mask | upper.str.contains(pattern, na=False)
    return df[mask].copy()


def _largest_cluster_mask(df, lat_col="latitude", lon_col="longitude", eps=0.015, min_samples=40):
    if len(df) < min_samples:
        return np.ones(len(df), dtype=bool)

    coords = df[[lon_col, lat_col]].dropna().to_numpy()
    if len(coords) < min_samples:
        return np.ones(len(df), dtype=bool)

    clustering = DBSCAN(eps=eps, min_samples=min_samples, algorithm="ball_tree")
    labels = clustering.fit_predict(coords)

    valid_labels = labels[labels != -1]
    if len(valid_labels) == 0:
        return np.ones(len(df), dtype=bool)

    largest_label = pd.Series(valid_labels).value_counts().idxmax()
    mask_valid = labels == largest_label

    full_mask = np.zeros(len(df), dtype=bool)
    valid_positions = df[[lon_col, lat_col]].dropna().index.to_numpy()
    index_to_position = {index: pos for pos, index in enumerate(df.index)}
    for idx, keep in zip(valid_positions, mask_valid):
        full_mask[index_to_position[idx]] = keep
    return full_mask


def filter_fortaleza_bbox(df, lat_col="latitude", lon_col="longitude", keep_main_cluster=True):
    filtered = df[
        (df[lat_col] >= FORTALEZA_LAT_RANGE[0])
        & (df[lat_col] <= FORTALEZA_LAT_RANGE[1])
        & (df[lon_col] >= FORTALEZA_LON_RANGE[0])
        & (df[lon_col] <= FORTALEZA_LON_RANGE[1])
    ].copy()

    if keep_main_cluster and not filtered.empty:
        mask = _largest_cluster_mask(filtered, lat_col=lat_col, lon_col=lon_col)
        filtered = filtered.loc[mask].copy()

    return filtered


def compute_bbox(df, lat_col="latitude", lon_col="longitude"):
    min_lon = df[lon_col].min()
    max_lon = df[lon_col].max()
    min_lat = df[lat_col].min()
    max_lat = df[lat_col].max()
    return (min_lon, min_lat, max_lon, max_lat)


def _largest_polygon(geometry):
    if geometry is None or geometry.is_empty:
        return geometry

    if isinstance(geometry, Polygon):
        return geometry

    if isinstance(geometry, MultiPolygon):
        polygons = [geom for geom in geometry.geoms if not geom.is_empty]
        return max(polygons, key=lambda geom: geom.area) if polygons else geometry

    if isinstance(geometry, GeometryCollection):
        polygons = [geom for geom in geometry.geoms if isinstance(geom, Polygon) and not geom.is_empty]
        return max(polygons, key=lambda geom: geom.area) if polygons else geometry

    return geometry


def build_study_area(df, lat_col="latitude", lon_col="longitude", concavity=0.12, buffer_ratio=0.008, min_buffer=0.001):
    coords = df[[lon_col, lat_col]].dropna().to_numpy()
    if len(coords) < 3:
        raise ValueError("Pontos insuficientes para estimar a área de estudo.")

    points = [Point(lon, lat) for lon, lat in coords]
    multipoint = MultiPoint(points)
    area = multipoint.convex_hull

    if shapely_concave_hull is not None and len(coords) >= 4:
        try:
            area = shapely_concave_hull(multipoint, ratio=concavity, allow_holes=False)
        except TypeError:
            area = shapely_concave_hull(multipoint, concavity, False)
        except Exception:
            area = multipoint.convex_hull

    area = _largest_polygon(area)

    if area.geom_type not in {"Polygon", "MultiPolygon"}:
        area = multipoint.convex_hull

    min_lon, min_lat, max_lon, max_lat = compute_bbox(df, lat_col=lat_col, lon_col=lon_col)
    span = max(max_lon - min_lon, max_lat - min_lat)
    buffer_distance = max(span * buffer_ratio, min_buffer)

    area = area.buffer(buffer_distance).buffer(0)
    area = _largest_polygon(area)
    return area


def iter_polygon_parts(geometry):
    if geometry is None or geometry.is_empty:
        return

    if isinstance(geometry, Polygon):
        yield geometry
        return

    if isinstance(geometry, MultiPolygon):
        for geom in geometry.geoms:
            if not geom.is_empty:
                yield geom
        return

    if isinstance(geometry, GeometryCollection):
        for geom in geometry.geoms:
            if isinstance(geom, Polygon) and not geom.is_empty:
                yield geom
