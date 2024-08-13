


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import stats
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 读取数据
current_path = os.path.dirname(__file__)
path_file1 = os.path.join(current_path, '原始数据_示例数据青少年体质数据.xlsx')
df = pd.read_excel(path_file1)

# 三阶多项式拟合
z = np.polyfit(df['年龄'], df['身高cm'], 3)
p = np.poly1d(z)

# 计算R²值
y_pred = p(df['年龄'])
ss_tot = np.sum((df['身高cm'] - np.mean(df['身高cm']))**2)
ss_res = np.sum((df['身高cm'] - y_pred)**2)
r_squared = 1 - (ss_res / ss_tot)

# 计算残差
residuals = df['身高cm'] - y_pred

# 绘图
plt.figure(figsize=(12, 10))

# 原始数据和拟合曲线
plt.subplot(2, 1, 1)
plt.scatter(df['年龄'], df['身高cm'], alpha=0.5)
x_plot = np.linspace(df['年龄'].min(), df['年龄'].max(), 100)
y_plot = p(x_plot)
plt.plot(x_plot, y_plot, 'r-', label='3阶多项式拟合')

# 计算置信区间
n = len(df['年龄'])
m = len(p.coeffs)
dof = n - m
t = stats.t.ppf(0.975, dof)
resid = df['身高cm'] - p(df['年龄'])
chi = np.sum((resid / p(df['年龄']))**2)
s_err = np.sqrt(chi / dof)
conf = t * s_err * np.sqrt(1/n + (x_plot - np.mean(df['年龄']))**2 / np.sum((df['年龄'] - np.mean(df['年龄']))**2))

plt.fill_between(x_plot, y_plot + conf, y_plot - conf, alpha=0.3, label='95% 置信区间')
plt.title('青少年年龄与身高关系图 (3阶多项式拟合)')
plt.xlabel('年龄')
plt.ylabel('身高 (cm)')
plt.legend()

# 残差图
plt.subplot(2, 1, 2)
plt.scatter(df['年龄'], residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('残差图')
plt.xlabel('年龄')
plt.ylabel('残差')

plt.tight_layout()
plt.show()

print(f"样本数量: {n}")
print(f"R²值: {r_squared:.4f}")
print(f"多项式方程: y = {p}")
