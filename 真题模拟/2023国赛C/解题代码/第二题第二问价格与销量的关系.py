import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 生成模拟数据
np.random.seed(0)
prices = np.linspace(5, 15, 100)
sales = 1000 - 50 * prices + np.random.normal(0, 50, 100)

# 创建散点图
plt.figure(figsize=(10, 6))
plt.scatter(prices, sales, alpha=0.5)
plt.title('价格与销量关系')
plt.xlabel('价格')
plt.ylabel('销量')

# 添加趋势线
slope, intercept, r_value, p_value, std_err = stats.linregress(prices, sales)
line = slope * prices + intercept
plt.plot(prices, line, color='r', label=f'趋势线 (R² = {r_value**2:.2f})')

plt.legend()
plt.grid(True)
plt.show()

# 计算价格弹性
average_price = np.mean(prices)
average_sales = np.mean(sales)
price_elasticity = (slope * average_price) / average_sales
print(f"价格弹性: {price_elasticity:.2f}")
