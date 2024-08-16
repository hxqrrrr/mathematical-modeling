import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端，适合保存动画


class TabuSearch:
    def __init__(self, distances, cities_coords, tabu_tenure=10, max_iterations=100):
        self.distances = distances
        self.cities_coords = cities_coords
        self.n_cities = len(distances)
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.tabu_list = {}
        self.best_solution = None
        self.best_distance = float('inf')
        self.solutions_history = []

    def initial_solution(self):
        return random.sample(range(self.n_cities), self.n_cities)

    def calculate_total_distance(self, solution):
        return sum(self.distances[solution[i - 1]][solution[i]] for i in range(self.n_cities))

    def get_neighbors(self, solution):
        neighbors = []
        for i in range(1, self.n_cities - 1):
            for j in range(i + 1, self.n_cities):
                neighbor = solution.copy()
                neighbor[i:j] = reversed(neighbor[i:j])
                neighbors.append(neighbor)
        return neighbors

    def run(self):
        current_solution = self.initial_solution()
        self.best_solution = current_solution
        self.best_distance = self.calculate_total_distance(self.best_solution)

        for iteration in range(self.max_iterations):
            neighbors = self.get_neighbors(current_solution)
            best_neighbor = None
            best_neighbor_distance = float('inf')

            for neighbor in neighbors:
                distance = self.calculate_total_distance(neighbor)
                if distance < best_neighbor_distance:
                    if tuple(neighbor) not in self.tabu_list or distance < self.best_distance:
                        best_neighbor = neighbor
                        best_neighbor_distance = distance

            current_solution = best_neighbor
            self.tabu_list[tuple(current_solution)] = self.tabu_tenure

            if best_neighbor_distance < self.best_distance:
                self.best_solution = current_solution
                self.best_distance = best_neighbor_distance

            # 记录每次迭代的最佳解
            self.solutions_history.append((self.best_solution.copy(), self.best_distance))

            # 更新禁忌列表
            for solution in list(self.tabu_list.keys()):
                self.tabu_list[solution] -= 1
                if self.tabu_list[solution] == 0:
                    del self.tabu_list[solution]

            # 打印进度
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Best distance: {self.best_distance:.2f}")

        return self.best_solution, self.best_distance


def visualize_tsp(cities_coords, solutions_history):
    fig, ax = plt.subplots(figsize=(10, 6))

    x = [city[0] for city in cities_coords]
    y = [city[1] for city in cities_coords]

    scatter = ax.scatter(x, y, c='red', s=50)
    line, = ax.plot([], [], c='blue', linewidth=2)
    title = ax.set_title("Iteration: 0, Distance: 0")

    ax.set_xlim(min(x) - 1, max(x) + 1)
    ax.set_ylim(min(y) - 1, max(y) + 1)

    def update(frame):
        solution, distance = solutions_history[frame]
        path_x = [cities_coords[i][0] for i in solution] + [cities_coords[solution[0]][0]]
        path_y = [cities_coords[i][1] for i in solution] + [cities_coords[solution[0]][1]]
        line.set_data(path_x, path_y)
        title.set_text(f"Iteration: {frame}, Distance: {distance:.2f}")
        return line, title

    anim = FuncAnimation(fig, update, frames=len(solutions_history), interval=200, repeat=False, blit=True)
    return anim


# 主程序
if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    random.seed(42)

    # 问题设置
    n_cities = 20
    cities_coords = np.random.rand(n_cities, 2) * 100  # 随机生成城市坐标
    distances = np.zeros((n_cities, n_cities))

    for i in range(n_cities):
        for j in range(n_cities):
            distances[i][j] = np.linalg.norm(cities_coords[i] - cities_coords[j])

    # 运行禁忌搜索
    ts = TabuSearch(distances, cities_coords, tabu_tenure=10, max_iterations=100)
    best_solution, best_distance = ts.run()

    print(f"Best solution: {best_solution}")
    print(f"Best distance: {best_distance:.2f}")

    # 创建并保存动画
    anim = visualize_tsp(cities_coords, ts.solutions_history)
    anim.save('tsp_tabu_search.gif', writer='pillow', fps=5)
    print("动画已保存为 'tsp_tabu_search.gif'")

    # 绘制最终结果
    plt.figure(figsize=(10, 6))
    x = [cities_coords[i][0] for i in best_solution + [best_solution[0]]]
    y = [cities_coords[i][1] for i in best_solution + [best_solution[0]]]
    plt.plot(x, y, 'bo-')
    plt.scatter(x, y, c='red', s=50)
    plt.title(f"Final TSP Solution - Distance: {best_distance:.2f}")
    plt.savefig('tsp_final_solution.png')
    print("最终解决方案图已保存为 'tsp_final_solution.png'")
