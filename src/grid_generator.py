import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree

class HexagonalGrid:
    def __init__(self, bbox, dx=0.0, dy=0.0, theta=0.0, R=0.01):
        """
        bbox: tuple (min_lon, min_lat, max_lon, max_lat)
        dx, dy: deslocamento horizontal e vertical (frações de R)
        theta: ângulo de rotação em radianos
        R: raio do hexágono (distância do centro a um vértice)
        """
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = bbox
        self.dx = dx * R
        self.dy = dy * R
        self.theta = theta
        self.R = R
        self.hexagons = []
        self.tree = None
        self._generate_grid()

    def _rotate_point(self, x, y, cx, cy, angle):
        """Rotaciona o ponto (x,y) ao redor de (cx, cy) por um determinado ângulo."""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        nx = cos_a * (x - cx) - sin_a * (y - cy) + cx
        ny = sin_a * (x - cx) + cos_a * (y - cy) + cy
        return nx, ny

    def _generate_grid(self):
        # Dimensões geométricas do hexágono regular flat-topped
        w = 2 * self.R
        h = np.sqrt(3) * self.R
        
        # Margem para garantir cobertura com deslocamentos/rotações
        margin = self.R * 3
        x_min = self.min_lon - margin
        x_max = self.max_lon + margin
        y_min = self.min_lat - margin
        y_max = self.max_lat + margin
        
        # Centróide central da bbox para rotação estável
        cx = (self.min_lon + self.max_lon) / 2
        cy = (self.min_lat + self.max_lat) / 2
        
        # Geração dos centros dos hexágonos em coordenadas locais pré-rotação e pré-deslocamento
        cols = int(np.ceil((x_max - x_min) / (w * 0.75)))
        rows = int(np.ceil((y_max - y_min) / h))
        
        polygons = []
        hex_id = 0
        
        # Definição básica de um hexágono regular centrado na origem (flat-topped)
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        local_vertices_x = self.R * np.cos(angles)
        local_vertices_y = self.R * np.sin(angles)
        
        for c in range(cols):
            x_center = x_min + c * (w * 0.75) + self.dx
            for r in range(rows):
                # Deslocamento vertical alternado para empacotamento hexagonal
                y_center = y_min + r * h + self.dy
                if c % 2 == 1:
                    y_center += h / 2
                
                # Gerar vértices, aplicar rotação sobre o centróide central (cx, cy)
                vertices = []
                for vx, vy in zip(local_vertices_x, local_vertices_y):
                    rx, ry = self._rotate_point(x_center + vx, y_center + vy, cx, cy, self.theta)
                    vertices.append((rx, ry))
                
                poly = Polygon(vertices)
                # Adiciona somente se interceptar a caixa delimitadora expandida
                polygons.append(poly)
                
        self.hexagons = polygons
        # Inicializar a STRtree para busca rápida
        self.tree = STRtree(self.hexagons)

    def assign_points(self, df, lat_col='latitude', lon_col='longitude'):
        """
        Retorna uma série com o índice do hexágono associado a cada registro no dataframe.
        Caso o ponto não caia em nenhum hexágono, retorna -1.
        """
        points = [Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])]
        
        # Buscar os índices correspondentes usando a STRtree
        indices = self.tree.query(points, predicate='within')
        
        # Criar vetor de atribuição
        assignment = np.full(len(df), -1, dtype=int)
        
        # indices[0] é o índice do ponto original, indices[1] é o índice do hexágono
        if len(indices) > 0:
            point_indices, hex_indices = indices
            assignment[point_indices] = hex_indices
            
        return assignment
