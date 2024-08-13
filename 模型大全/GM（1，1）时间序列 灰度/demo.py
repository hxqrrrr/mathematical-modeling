import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


import numpy as np
import matplotlib.pyplot as plt

# 原始数据
years = np.array([2016, 2017, 2018, 2019])
population = np.array([50, 58, 67, 78])

# 步骤1: 累加生成
X1 = np.cumsum(population)

# 步骤2: 生成矩阵B和向量Y
n = len(X1)
B = np.zeros((n-1, 2))
for i in range(n-1):
    B[i][0] = -(X1[i] + X1[i+1]) / 2
    B[i][1] = 1
Y = population[1:]

# 步骤3: 计算参数a和u
a, u = np.linalg.lstsq(B, Y, rcond=None)[0]

# 步骤4: 建立时间响应序列
def model(k):
    return (population[0] - u/a) * np.exp(-a*(k-1)) + u/a

# 预测未来3年
future_years = np.array([2020, 2021, 2022])
predictions = np.array([model(k) for k in range(n+1, n+4)])

# 打印预测结果
for year, pred in zip(future_years, predictions):
    print(f"{year}年预测人口: {pred:.2f}万")

# 绘制原始数据和预测数据
plt.figure(figsize=(10, 6))
plt.plot(years, population, 'bo-', label='实际数据')
plt.plot(future_years, predictions, 'ro--', label='预测数据')
plt.xlabel('年份')
plt.ylabel('人口 (万人)')
plt.title('GM(1,1)模型人口增长预测')
plt.legend()
plt.grid(True)

# 显示所有数据点的值
for x, y in zip(years, population):
    plt.text(x, y, f'{y}', ha='center', va='bottom')
for x, y in zip(future_years, predictions):
    plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')

plt.show()
