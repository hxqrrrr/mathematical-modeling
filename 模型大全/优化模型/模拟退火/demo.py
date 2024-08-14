import math
import random


def objective_function(x):
    """
    目标函数，这里我们尝试最小化 f(x) = x^2
    """
    return x * x


def simulated_annealing(initial_state, temperature, cooling_rate, num_iterations):
    current_state = initial_state
    current_energy = objective_function(current_state)
    best_state = current_state
    best_energy = current_energy

    for i in range(num_iterations):
        # 生成新的候选解
        neighbor = current_state + random.uniform(-1, 1)

        # 计算新解的能量
        neighbor_energy = objective_function(neighbor)

        # 计算能量差
        energy_diff = neighbor_energy - current_energy

        # 判断是否接受新解
        if energy_diff < 0 or random.random() < math.exp(-energy_diff / temperature):
            current_state = neighbor
            current_energy = neighbor_energy

            # 更新最优解
            if current_energy < best_energy:
                best_state = current_state
                best_energy = current_energy

        # 降温
        temperature *= cooling_rate

        # 打印当前迭代的结果
        if (i + 1) % 100 == 0:
            print(f"Iteration {i + 1}: Best State = {best_state:.4f}, Best Energy = {best_energy:.4f}")

    return best_state, best_energy


# 设置参数
initial_state = 10.0  # 初始状态
initial_temperature = 100.0  # 初始温度
cooling_rate = 0.995  # 冷却率
num_iterations = 1000  # 迭代次数

# 运行模拟退火算法
best_solution, best_value = simulated_annealing(initial_state, initial_temperature, cooling_rate, num_iterations)

print(f"\nFinal Result:")
print(f"Best Solution: x = {best_solution:.6f}")
print(f"Best Value: f(x) = {best_value:.6f}")
