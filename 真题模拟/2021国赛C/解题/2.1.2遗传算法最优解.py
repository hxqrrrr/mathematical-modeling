import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

class OptimizedBinaryArrayGA:

    def __init__(self, objective_func, constraint_func, array_length, pop_size=100, generations=100, mutation_rate=0.01,
                 crossover_rate=0.8):
        self.objective_func = objective_func
        self.constraint_func = constraint_func
        self.array_length = array_length
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_objective = float('inf')
        self.objective_history = []
        self.constraint_history = []
        self.total_value_history = []  # 新增：用于记录total_value历史

    def initialize_population(self):
        self.population = np.random.randint(2, size=(self.pop_size, self.array_length))
        self.fitness = np.zeros(self.pop_size)

    def evaluate_fitness(self):
        best_constraint = float('inf')
        best_total_value = 0
        for i, ind in enumerate(self.population):
            objective_value = self.objective_func(ind)
            constraint_violation, total_value = self.constraint_func(ind)

            if constraint_violation > 0:
                self.fitness[i] = objective_value + 1000 * constraint_violation
            else:
                self.fitness[i] = objective_value

            if constraint_violation < best_constraint:
                best_constraint = constraint_violation
                best_total_value = total_value

        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_objective and self.constraint_func(self.population[best_idx])[0] == 0:
            self.best_solution = self.population[best_idx].copy()
            self.best_objective = self.objective_func(self.best_solution)

        self.constraint_history.append(best_constraint)
        self.total_value_history.append(best_total_value)

    def select_parents(self):
        tournament_size = 3
        selected = np.zeros((self.pop_size, self.array_length), dtype=int)
        for i in range(self.pop_size):
            tournament = np.random.choice(self.pop_size, tournament_size, replace=False)
            winner = tournament[np.argmin(self.fitness[tournament])]
            selected[i] = self.population[winner]
        return selected

    def crossover(self, parents):
        children = np.copy(parents)
        for i in range(0, self.pop_size - 1, 2):
            if np.random.random() < self.crossover_rate:
                crossover_point = np.random.randint(1, self.array_length)
                children[i, crossover_point:] = parents[i + 1, crossover_point:]
                children[i + 1, crossover_point:] = parents[i, crossover_point:]
        return children

    def mutate(self, children):
        mutation_mask = np.random.random(children.shape) < self.mutation_rate
        children[mutation_mask] = 1 - children[mutation_mask]
        return children

    def optimize(self):
        self.initialize_population()

        for generation in range(self.generations):
            self.evaluate_fitness()
            self.objective_history.append(self.best_objective)

            if generation % 10 == 0:
                print(f"Generation {generation}: Best objective = {self.best_objective:.4f}, "
                      f"Best constraint = {self.constraint_history[-1]:.4f}, "
                      f"Total value = {self.total_value_history[-1]:.4f}")

            parents = self.select_parents()
            children = self.crossover(parents)
            self.population = self.mutate(children)

        self.evaluate_fitness()
        print(f"\nOptimization completed:")
        print(f"Best objective value: {self.best_objective:.4f}")
        print(f"Final constraint value: {self.constraint_history[-1]:.4f}")
        print(f"Final total value: {self.total_value_history[-1]:.4f}")

        return self.best_solution

    def plot_history(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

        ax1.plot(self.objective_history)
        ax1.set_title('Best Objective Value over Generations')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Best Objective Value')

        ax2.plot(self.constraint_history)
        ax2.set_title('Best Constraint Value over Generations')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Best Constraint Value')

        ax3.plot(self.total_value_history)
        ax3.set_title('Total Value over Generations')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Total Value')

        plt.tight_layout()
        plt.show()

    def plot_objective_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.objective_history)
        plt.title('Best Objective Value over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Best Objective Value')
        plt.show()

# 读取数据
file_path = '附件1 近5年402家供应商的相关数据.xlsx'

data = pd.read_excel(file_path,sheet_name=0)
df = pd.DataFrame({
    '供应商ID': data['供应商ID'],
    'sum':data.iloc[:,2:].sum(axis=1),
    'mean':data.iloc[:,2:].mean(axis=1),
    '材料分类':data['材料分类'],
})

df['原料还原值']=df['材料分类'].map({'A':1/0.6,'B':1/0.66,'C':1/0.72})*df['mean']
df = df[(df['原料还原值']>100)]



# 使用示例
def objective_function(array):
    return np.sum(array)  # 最小化1的数量


def constraint_function(array):
    total_value = (df['原料还原值'] * array).sum()
    return max(0, 28200 - total_value), total_value  # 返回约束违反值和total_value
# 创建并运行遗传算法
ga = OptimizedBinaryArrayGA(objective_function, constraint_function, array_length=df.shape[0], pop_size=500, generations=1000)
optimized_array = ga.optimize()
pd.DataFrame(optimized_array).to_csv('38家核心供应商.csv', index=False)

# 绘制目标值、约束值和total_value历史
ga.plot_history()

# 打印最终结果
selected_suppliers = df[optimized_array == 1]
print(f"\n选中的供应商数量: {sum(optimized_array)}")
print(f"总原料还原值: {(df['原料还原值'] * optimized_array).sum():.2f}")
print("选中的供应商ID:")
print(selected_suppliers['供应商ID'].tolist())