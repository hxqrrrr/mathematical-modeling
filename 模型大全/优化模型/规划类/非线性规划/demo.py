import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 定义目标函数
def objective(x):
    return (x[0] - 1)**2 + (x[1] - 2.5)**2

# 定义约束条件
def constraint1(x):
    return 50 - x[0]**2 - x[1]**2

def constraint2(x):
    return x[1] + x[0] - 1

# 设置约束条件
cons = ({'type': 'ineq', 'fun': constraint1},
        {'type': 'ineq', 'fun': constraint2})

# 设置变量的边界
bnds = ((0, None), (0, None))

# 设置初始猜测
x0 = [0, 0]

# 使用SLSQP方法求解
solution = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)

# 打印结果
print('最优解:')
print('x =', solution.x[0])
print('y =', solution.x[1])
print('目标函数值 =', solution.fun)
print('优化是否成功:', solution.success)
print('迭代次数:', solution.nit)

# 可视化
x = np.linspace(0, 8, 100)
y = np.linspace(0, 8, 100)
X, Y = np.meshgrid(x, y)
Z = (X - 1)**2 + (Y - 2.5)**2

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='目标函数值')

# 绘制约束条件
plt.contour(X, Y, X**2 + Y**2, levels=[50], colors='r', linestyles='dashed')
plt.contour(X, Y, Y + X, levels=[1], colors='g', linestyles='dashed')

# 绘制最优解
plt.plot(solution.x[0], solution.x[1], 'ro', markersize=10, label='最优解')

plt.xlabel('x')
plt.ylabel('y')
plt.title('非线性规划问题可视化')
plt.legend()
plt.grid(True)
plt.show()
