import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 设置随机种子以保证结果可重现
np.random.seed(42)

# 定义状态和转移概率矩阵
states = ["晴天", "多云", "雨天"]
transition_matrix = np.array([
    [0.7, 0.2, 0.1],  # 晴天转移概率
    [0.3, 0.5, 0.2],  # 多云转移概率
    [0.2, 0.3, 0.5]  # 雨天转移概率
])


def simulate_weather_vectorized(days, initial_state):
    # 初始化状态
    state_indices = np.zeros(days, dtype=int)
    state_indices[0] = states.index(initial_state)

    # 生成随机数
    random_numbers = np.random.random(days - 1)

    # 向量化的状态转移
    for i in range(1, days):
        state_indices[i] = np.searchsorted(
            np.cumsum(transition_matrix[state_indices[i - 1]]),
            random_numbers[i - 1]
        )

    return np.array(states)[state_indices]


# 运行模拟
days = 365
initial_state = "晴天"
weather = simulate_weather_vectorized(days, initial_state)

# 创建日期范围
date_range = pd.date_range(start='2024-01-01', periods=days)

# 创建DataFrame
df = pd.DataFrame({'date': date_range, 'weather': weather})

# 统计结果
weather_counts = df['weather'].value_counts()
print("天气统计:")
print(weather_counts)
print(f"\n总天数: {days}")

# 可视化结果
plt.figure(figsize=(15, 10))

# 绘制天气状态变化
plt.subplot(2, 1, 1)
sns.scatterplot(data=df, x='date', y='weather', hue='weather', palette='viridis', s=50)
plt.title('一年天气状态变化')
plt.xlabel('日期')
plt.ylabel('天气状态')
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

# 绘制天气状态分布
plt.subplot(2, 1, 2)
sns.barplot(x=weather_counts.index, y=weather_counts.values, hue=weather_counts.index, palette='viridis', legend=False)

plt.title('天气状态分布')
plt.xlabel('天气状态')
plt.ylabel('天数')

plt.tight_layout()
plt.show()

# 计算每月天气状态分布
df['month'] = df['date'].dt.month
monthly_weather = df.groupby('month')['weather'].value_counts().unstack(fill_value=0)

# 绘制每月天气状态分布热图
plt.figure(figsize=(12, 8))
sns.heatmap(monthly_weather, annot=True, fmt='d', cmap='YlGnBu')
plt.title('每月天气状态分布')
plt.xlabel('天气状态')
plt.ylabel('月份')
plt.show()
