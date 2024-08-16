import numpy as np

class AntColonyOptimization:
    def __init__(self, distances, n_ants, n_iterations, alpha, beta, evaporation_rate, Q):
        self.distances = distances
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # 信息素重要程度因子
        self.beta = beta    # 启发式因子重要程度因子
        self.evaporation_rate = evaporation_rate  # 信息素蒸发系数
        self.Q = Q  # 信息素增加强度系数
        self.n_cities = len(distances)
        self.pheromones = np.ones((self.n_cities, self.n_cities))

    def run(self):
        best_path = None
        best_path_length = float('inf')
        for iteration in range(self.n_iterations):
            paths = self.construct_paths()
            self.update_pheromones(paths)
            iteration_best_path = min(paths, key=lambda x: self.path_length(x))
            iteration_best_path_length = self.path_length(iteration_best_path)
            if iteration_best_path_length < best_path_length:
                best_path = iteration_best_path
                best_path_length = iteration_best_path_length
            print(f"Iteration {iteration + 1}, Best Length: {best_path_length}")
        return best_path, best_path_length

    def construct_paths(self):
        paths = []
        for ant in range(self.n_ants):
            path = self.construct_path_for_ant()
            paths.append(path)
        return paths

    def construct_path_for_ant(self):
        start_city = np.random.randint(self.n_cities)
        path = [start_city]
        available_cities = set(range(self.n_cities)) - {start_city}
        while available_cities:
            next_city = self.choose_next_city(path[-1], available_cities)
            path.append(next_city)
            available_cities.remove(next_city)
        return path

    def choose_next_city(self, current_city, available_cities):
        probabilities = []
        for city in available_cities:
            pheromone = self.pheromones[current_city][city]
            distance = self.distances[current_city][city]
            probability = (pheromone ** self.alpha) * ((1.0 / distance) ** self.beta)
            probabilities.append(probability)
        probabilities = np.array(probabilities) / sum(probabilities)
        return np.random.choice(list(available_cities), p=probabilities)

    def update_pheromones(self, paths):
        self.pheromones *= (1 - self.evaporation_rate)
        for path in paths:
            path_length = self.path_length(path)
            for i in range(len(path) - 1):
                self.pheromones[path[i]][path[i+1]] += self.Q / path_length
            self.pheromones[path[-1]][path[0]] += self.Q / path_length

    def path_length(self, path):
        return sum(self.distances[path[i]][path[i+1]] for i in range(len(path) - 1)) + self.distances[path[-1]][path[0]]

# 使用示例
distances = np.array([
    [0, 2, 9, 10],
    [1, 0, 6, 4],
    [15, 7, 0, 8],
    [6, 3, 12, 0]
])

aco = AntColonyOptimization(distances, n_ants=10, n_iterations=100, alpha=1, beta=2, evaporation_rate=0.1, Q=1)
best_path, best_path_length = aco.run()

print(f"Best Path: {best_path}")
print(f"Best Path Length: {best_path_length}")
