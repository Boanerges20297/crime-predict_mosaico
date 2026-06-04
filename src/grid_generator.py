import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.strtree import STRtree


class HexagonalGrid:
    def __init__(self, bbox, dx=0.0, dy=0.0, theta=0.0, R=0.01, study_area=None):
        """
        bbox: tuple (min_lon, min_lat, max_lon, max_lat)
        dx, dy: deslocamento horizontal e vertical (frações de R)
        theta: ângulo de rotação em radianos
        R: raio do hexágono (distância do centro a um vértice)
        study_area: polígono da área de estudo para recorte da malha
        """
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = bbox
        self.dx = dx * R
        self.dy = dy * R
        self.theta = theta
        self.R = R
        self.study_area = study_area
        self.coverage_area = study_area if study_area is not None else box(self.min_lon, self.min_lat, self.max_lon, self.max_lat)
        self.hexagons = []
        self.display_hexagons = []
        self.tree = None
        self._generate_grid()

    def _rotate_point(self, x, y, cx, cy, angle):
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        nx = cos_a * (x - cx) - sin_a * (y - cy) + cx
        ny = sin_a * (x - cx) + cos_a * (y - cy) + cy
        return nx, ny

    def _generate_grid(self):
        w = 2 * self.R
        h = np.sqrt(3) * self.R

        margin = self.R * 3
        x_min = self.min_lon - margin
        x_max = self.max_lon + margin
        y_min = self.min_lat - margin
        y_max = self.max_lat + margin

        cx = (self.min_lon + self.max_lon) / 2
        cy = (self.min_lat + self.max_lat) / 2

        cols = int(np.ceil((x_max - x_min) / (w * 0.75)))
        rows = int(np.ceil((y_max - y_min) / h))

        polygons = []
        display_polygons = []

        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        local_vertices_x = self.R * np.cos(angles)
        local_vertices_y = self.R * np.sin(angles)

        for col in range(cols):
            x_center = x_min + col * (w * 0.75) + self.dx
            for row in range(rows):
                y_center = y_min + row * h + self.dy
                if col % 2 == 1:
                    y_center += h / 2

                vertices = []
                for vx, vy in zip(local_vertices_x, local_vertices_y):
                    rx, ry = self._rotate_point(x_center + vx, y_center + vy, cx, cy, self.theta)
                    vertices.append((rx, ry))

                poly = Polygon(vertices)
                if not poly.intersects(self.coverage_area):
                    continue

                clipped_poly = poly.intersection(self.coverage_area)
                if clipped_poly.is_empty:
                    continue

                polygons.append(poly)
                display_polygons.append(clipped_poly)

        self.hexagons = polygons
        self.display_hexagons = display_polygons
        self.tree = STRtree(self.hexagons)

    def assign_points(self, df, lat_col="latitude", lon_col="longitude"):
        """
        Retorna um vetor com o índice do hexágono associado a cada registro.
        Caso o ponto não caia em nenhum hexágono, retorna -1.
        """
        points = [Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])]
        indices = self.tree.query(points, predicate="within")

        assignment = np.full(len(df), -1, dtype=int)
        if len(indices) > 0:
            point_indices, hex_indices = indices
            assignment[point_indices] = hex_indices

        return assignment
