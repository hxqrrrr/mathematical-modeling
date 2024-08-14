import math
import random
import matplotlib.pyplot as plt

# 定义问题参数
PROFIT_GOAL = 2400
PRODUCTION_GOAL = 40
WORK_HOURS_GOAL = 160


# 定义目标函数
def objective_function(x1, x2):
    profit = 40 * x1 + 30 * x2
    production = x1 + x2
    work_hours = 4 * x1 + 3 * x2

    # 计算每个目标的偏差
    profit_dev = abs(profit - PROFIT_GOAL)
    production_dev = abs(production - PRODUCTION_GOAL)
    work_hours_dev = abs(work_hours - WORK_HOURS_GOAL)

    # 返回总偏差（这里我们给予每个目标相等的权重）
    return profit_dev + production_dev + work_hours_dev


# 模拟退火算法
def simulated_annealing(initial_temp, cooling_rate, num_iterations):
    # 初始解
    current_x1 = random.uniform(0, 60)
    current_x2 = random.uniform(0, 60)
    current_cost = objective_function(current_x1, current_x2)

    best_x1, best_x2 = current_x1, current_x2
    best_cost = current_cost

    temperature = initial_temp

    cost_history = []

    for i in range(num_iterations):
        # 生成新的候选解
        new_x1 = current_x1 + random.uniform(-5, 5)
        new_x2 = current_x2 + random.uniform(-5, 5)

        # 确保新解在有效范围内
        new_x1 = max(0, min(60, new_x1))
        new_x2 = max(0, min(60, new_x2))

        new_cost = objective_function(new_x1, new_x2)

        # 计算成本差
        cost_diff = new_cost - current_cost

        # 决定是否接受新解
        if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
            current_x1, current_x2 = new_x1, new_x2
            current_cost = new_cost

            if current_cost < best_cost:
                best_x1, best_x2 = current_x1, current_x2
                best_cost = current_cost

        # 降温
        temperature *= cooling_rate

        cost_history.append(best_cost)

    return best_x1, best_x2, best_cost, cost_history


# 运行模拟退火算法
initial_temp = 1000
cooling_rate = 0.995
num_iterations = 1000

best_x1, best_x2, best_cost, cost_history = simulated_annealing(initial_temp, cooling_rate, num_iterations)

print(f"Best solution: x1 = {best_x1:.2f}, x2 = {best_x2:.2f}")
print(f"Best cost: {best_cost:.2f}")

# 可视化结果
plt.figure(figsize=(12, 5))

# 绘制成本历史
plt.subplot(1, 2, 1)
plt.plot(cost_history)
plt.title('Cost History')
plt.xlabel('Iteration')
plt.ylabel('Cost')

# 绘制最终解
plt.subplot(1, 2, 2)
x = range(61)
y1 = [(PROFIT_GOAL - 40 * i) / 30 for i in x]
y2 = [PRODUCTION_GOAL - i for i in x]
y3 = [(WORK_HOURS_GOAL - 4 * i) / 3 for i in x]

plt.plot(x, y1, label='Profit Goal')
plt.plot(x, y2, label='Production Goal')
plt.plot(x, y3, label='Work Hours Goal')
plt.plot(best_x1, best_x2, 'ro', markersize=10, label='Best Solution')

plt.xlim(0, 60)
plt.ylim(0, 60)
plt.xlabel('Product 1')
plt.ylabel('Product 2')
plt.title('Goal Programming with Simulated Annealing')
plt.legend()

plt.tight_layout()
plt.show()
