import numpy as np
import matplotlib.pyplot as plt

# 定义约束函数
def constraint1(x):
    return 100 - 5*x

def constraint2(x):
    return 80 - x

def constraint3(x):
    return 40 * np.ones_like(x)

# 创建图形
plt.figure(figsize=(12, 9))

# 绘制约束线
x = np.linspace(0, 60, 100)
plt.plot(x, constraint1(x), label='2x + y ≤ 100', color='blue')
plt.plot(x, constraint2(x), label='x + y ≤ 80', color='red')
plt.plot(x, constraint3(x), label='x ≤ 40', color='green')

# 填充可行区域
y1 = np.minimum(constraint1(x), constraint2(x))
y2 = np.minimum(y1, constraint3(x))
plt.fill_between(x, 0, y2, where=(y2 >= 0), alpha=0.2, color='gray', label='可行域')

# 标记最优解
optimal_x, optimal_y = 20, 60
plt.plot(optimal_x, optimal_y, 'r*', markersize=15, label='最优解 (20, 60)')

# 设置图形属性
plt.xlim(0, 60)
plt.ylim(0, 100)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('线性规划问题可视化', fontsize=16)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# 添加目标函数等值线
x, y = np.meshgrid(np.linspace(0, 60, 100), np.linspace(0, 100, 100))
z = 3*x + 2*y
contour = plt.contour(x, y, z, levels=10, alpha=0.5, colors='purple', linestyles='dashed')
plt.clabel(contour, inline=True, fontsize=8, fmt='%1.0f')

# 添加注释
plt.annotate('目标函数: z = 3x + 2y', xy=(30, 90), xytext=(35, 95),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

plt.tight_layout()
plt.show()
