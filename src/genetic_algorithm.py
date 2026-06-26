import math
import time

import numpy as np

from grid_generator import HexagonalGrid
from predictor import evaluate_model, prepare_weekly_series


def compute_discretization_limits(study_area, bbox, target_hex_count, min_hex_count_ratio=0.75):
    min_hex_count = max(1, int(round(target_hex_count * min_hex_count_ratio)))

    if study_area is not None and not study_area.is_empty:
        coverage_area = float(study_area.area)
    else:
        min_lon, min_lat, max_lon, max_lat = bbox
        coverage_area = float(max(max_lon - min_lon, 0.0) * max(max_lat - min_lat, 0.0))

    if coverage_area <= 0:
        max_radius = 0.03
    else:
        hex_area_constant = 3 * math.sqrt(3) / 2
        discretization_radius = math.sqrt(coverage_area / (max(min_hex_count, 1) * hex_area_constant))
        max_radius = float(np.clip(discretization_radius, 0.0075, 0.03))

    return {
        "min_hex_count": min_hex_count,
        "min_hex_count_ratio": min_hex_count_ratio,
        "max_radius": max_radius,
    }


class GeneticAlgorithmHex:
    def __init__(
        self,
        df,
        bbox,
        study_area=None,
        pop_size=10,
        generations=5,
        mutation_rate=0.15,
        crossover_rate=0.8,
        seed=42,
        target_hex_count=250,
        hex_penalty_weight=6.0,
        min_hex_count_ratio=0.75,
        discretization_penalty_weight=8.0,
    ):
        self.df = df
        self.bbox = bbox
        self.study_area = study_area
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.seed = seed
        self.target_hex_count = target_hex_count
        self.hex_penalty_weight = hex_penalty_weight
        self.min_hex_count_ratio = min_hex_count_ratio
        self.discretization_penalty_weight = discretization_penalty_weight
        np.random.seed(seed)

        discretization_limits = compute_discretization_limits(
            study_area=self.study_area,
            bbox=self.bbox,
            target_hex_count=self.target_hex_count,
            min_hex_count_ratio=self.min_hex_count_ratio,
        )
        self.min_hex_count = discretization_limits["min_hex_count"]
        self.max_radius = discretization_limits["max_radius"]
        self.bounds = [
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, np.pi / 3),
            (0.0075, self.max_radius),
        ]
        self.population = self._init_population()

    def _init_population(self):
        population = []
        for _ in range(self.pop_size):
            individual = []
            for low, high in self.bounds:
                individual.append(np.random.uniform(low, high))
            population.append(np.array(individual))
        return population

    def _score_individual(self, individual):
        dx, dy, theta, radius = individual
        grid = HexagonalGrid(self.bbox, dx=dx, dy=dy, theta=theta, R=radius, study_area=self.study_area)

        df_temp = self.df.copy()
        df_temp["hex_id"] = grid.assign_points(df_temp)
        df_temp = df_temp[df_temp["hex_id"] != -1]

        if len(df_temp) < 100:
            return {
                "fitness": 1e-8,
                "mse": np.nan,
                "penalized_mse": np.inf,
                "hex_count": 0,
                "active_hex_count": 0,
                "discretization_gap": 1.0,
            }

        df_series = prepare_weekly_series(df_temp, region_col="hex_id", lags=3)
        mse = evaluate_model(df_series, region_col="hex_id", lags=3)
        if np.isnan(mse):
            return {
                "fitness": 1e-8,
                "mse": np.nan,
                "penalized_mse": np.inf,
                "hex_count": len(grid.display_hexagons),
                "active_hex_count": int(df_temp["hex_id"].nunique()),
                "discretization_gap": 1.0,
            }

        hex_count = len(grid.display_hexagons)
        active_hex_count = int(df_temp["hex_id"].nunique())
        hex_deviation = abs(hex_count - self.target_hex_count) / max(self.target_hex_count, 1)
        discretization_gap = max(self.min_hex_count - hex_count, 0) / max(self.min_hex_count, 1)
        penalized_mse = (
            mse
            + (self.hex_penalty_weight * hex_deviation)
            + (self.discretization_penalty_weight * discretization_gap)
        )
        fitness = 1.0 / (penalized_mse + 1e-8)

        return {
            "fitness": fitness,
            "mse": mse,
            "penalized_mse": penalized_mse,
            "hex_count": hex_count,
            "active_hex_count": active_hex_count,
            "discretization_gap": discretization_gap,
        }

    def fitness(self, individual):
        return self._score_individual(individual)["fitness"]

    def select(self, population, fitnesses):
        selected = []
        for _ in range(self.pop_size):
            candidate_indices = np.random.choice(self.pop_size, 3, replace=False)
            best_idx = candidate_indices[np.argmax([fitnesses[index] for index in candidate_indices])]
            selected.append(population[best_idx].copy())
        return selected

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            alpha = np.random.uniform(0.1, 0.9)
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2

            for index, (low, high) in enumerate(self.bounds):
                child1[index] = np.clip(child1[index], low, high)
                child2[index] = np.clip(child2[index], low, high)
            return child1, child2

        return parent1.copy(), parent2.copy()

    def mutate(self, individual, generation):
        mutation_strength = 0.1 * (1.0 - generation / self.generations)
        for index in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                low, high = self.bounds[index]
                noise = np.random.normal(0, mutation_strength * (high - low))
                individual[index] = np.clip(individual[index] + noise, low, high)
        return individual

    def run(self):
        best_individual = None
        best_fitness = -1.0

        print("Iniciando evolução...")
        print(
            f"Limite dinâmico do raio: R <= {self.max_radius:.6f} | "
            f"Piso de discretização: {self.min_hex_count} hexágonos"
        )
        for generation in range(self.generations):
            start = time.time()
            fitnesses = []

            for individual in self.population:
                score = self._score_individual(individual)
                fitness = score["fitness"]
                fitnesses.append(fitness)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()

            new_population = [best_individual.copy()]
            mating_pool = self.select(self.population, fitnesses)

            for index in range(1, self.pop_size, 2):
                parent1 = mating_pool[index - 1]
                parent2 = mating_pool[index] if index < self.pop_size else mating_pool[0]

                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child1, generation))
                if len(new_population) < self.pop_size:
                    new_population.append(self.mutate(child2, generation))

            self.population = new_population[: self.pop_size]

            best_score = self._score_individual(best_individual)
            elapsed = time.time() - start
            print(
                f"Geração {generation + 1}/{self.generations} - "
                f"EQM: {best_score['mse']:.4f} | "
                f"EQM penalizado: {best_score['penalized_mse']:.4f} | "
                f"Hex criados: {best_score['hex_count']} | "
                f"Gap discretização: {best_score['discretization_gap']:.3f} | "
                f"Tempo: {elapsed:.2f}s"
            )

        dx, dy, theta, radius = best_individual
        best_score = self._score_individual(best_individual)
        return {
            "dx": dx,
            "dy": dy,
            "theta": theta,
            "R": radius,
            "best_mse": best_score["mse"],
            "penalized_mse": best_score["penalized_mse"],
            "hex_count": best_score["hex_count"],
            "active_hex_count": best_score["active_hex_count"],
            "target_hex_count": self.target_hex_count,
            "min_hex_count": self.min_hex_count,
            "min_hex_count_ratio": self.min_hex_count_ratio,
            "max_radius": self.max_radius,
            "discretization_penalty_weight": self.discretization_penalty_weight,
        }
