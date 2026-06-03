import numpy as np
import pandas as pd
import time
from grid_generator import HexagonalGrid
from predictor import prepare_weekly_series, evaluate_model

class GeneticAlgorithmHex:
    def __init__(self, df, bbox, pop_size=10, generations=5, mutation_rate=0.15, crossover_rate=0.8, seed=42):
        """
        df: DataFrame limpo de crimes.
        bbox: Caixa delimitadora geográfica (min_lon, min_lat, max_lon, max_lat)
        """
        self.df = df
        self.bbox = bbox
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.seed = seed
        np.random.seed(seed)
        
        # Limites lógicos para os genes (dx, dy, theta, R)
        self.bounds = [
            (0.0, 1.0),      # dx (proporção do raio)
            (0.0, 1.0),      # dy (proporção do raio)
            (0.0, np.pi/3),  # theta (rotação local do hexágono)
            (0.005, 0.02)    # R (raio em graus, ~550m a ~2200m)
        ]
        self.population = self._init_population()

    def _init_population(self):
        pop = []
        for _ in range(self.pop_size):
            ind = []
            for (low, high) in self.bounds:
                ind.append(np.random.uniform(low, high))
            pop.append(np.array(ind))
        return pop

    def fitness(self, individual):
        """
        Aptidão do indivíduo. Quanto menor o EQM (MSE), melhor.
        Retornaremos (1.0 / (EQM + 1e-8)) como aptidão para maximizar.
        """
        dx, dy, theta, R = individual
        
        # Gerar a grade hexagonal e indexar pontos
        grid = HexagonalGrid(self.bbox, dx=dx, dy=dy, theta=theta, R=R)
        
        # Criar coluna temporária com os índices atribuídos
        df_temp = self.df.copy()
        df_temp['hex_id'] = grid.assign_points(df_temp)
        
        # Filtrar pontos fora de hexágonos (caso ocorra nas bordas devido à rotação extrema)
        df_temp = df_temp[df_temp['hex_id'] != -1]
        
        if len(df_temp) < 100:  # Penalidade se a grade falhar em cobrir os pontos
            return 1e-8
            
        # Preparar séries e avaliar
        df_series = prepare_weekly_series(df_temp, region_col='hex_id', lags=3)
        mse = evaluate_model(df_series, region_col='hex_id', lags=3)
        
        if np.isnan(mse):
            return 1e-8
            
        # Retorna aptidão inversa para maximização
        return 1.0 / (mse + 1e-8)

    def select(self, pop, fitnesses):
        # Seleção por Torneio (tamanho = 3)
        selected = []
        for _ in range(self.pop_size):
            cand_indices = np.random.choice(self.pop_size, 3, replace=False)
            best_idx = cand_indices[np.argmax([fitnesses[i] for i in cand_indices])]
            selected.append(pop[best_idx].copy())
        return selected

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            # Crossover Aritmético Simples
            alpha = np.random.uniform(0.1, 0.9)
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2
            
            # Garantir limites
            for i, (low, high) in enumerate(self.bounds):
                child1[i] = np.clip(child1[i], low, high)
                child2[i] = np.clip(child2[i], low, high)
            return child1, child2
        return parent1.copy(), parent2.copy()

    def mutate(self, individual, gen):
        # Mutação gaussiana adaptativa (decaimento com a geração)
        mutation_strength = 0.1 * (1.0 - gen / self.generations)
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                low, high = self.bounds[i]
                noise = np.random.normal(0, mutation_strength * (high - low))
                individual[i] = np.clip(individual[i] + noise, low, high)
        return individual

    def run(self):
        best_ind = None
        best_fitness = -1.0
        
        print("Iniciando evolução...")
        for g in range(self.generations):
            start = time.time()
            fitnesses = []
            
            # Avaliar população
            for idx, ind in enumerate(self.population):
                fit = self.fitness(ind)
                fitnesses.append(fit)
                if fit > best_fitness:
                    best_fitness = fit
                    best_ind = ind.copy()
            
            # Elitismo
            new_pop = [best_ind.copy()]
            
            # Seleção
            mating_pool = self.select(self.population, fitnesses)
            
            # Reprodução (Crossover e Mutação)
            for i in range(1, self.pop_size, 2):
                p1 = mating_pool[i-1]
                p2 = mating_pool[i] if i < self.pop_size else mating_pool[0]
                
                c1, c2 = self.crossover(p1, p2)
                new_pop.append(self.mutate(c1, g))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self.mutate(c2, g))
            
            self.population = new_pop[:self.pop_size]
            
            best_mse = 1.0 / best_fitness
            elapsed = time.time() - start
            print(f"Geração {g+1}/{self.generations} - Melhor EQM (MSE): {best_mse:.4f} | Tempo: {elapsed:.2f}s")
            
        # Extrair parâmetros ótimos do melhor indivíduo
        dx, dy, theta, R = best_ind
        return {
            'dx': dx,
            'dy': dy,
            'theta': theta,
            'R': R,
            'best_mse': 1.0 / best_fitness
        }
