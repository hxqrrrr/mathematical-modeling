import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from math import sqrt

# 设置参数
target_column = '辣椒类'



plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

parent_dir = os.path.dirname(os.getcwd())
path_file = os.path.join(parent_dir, "解题代码", "分类全销量按日.xlsx")

dates = pd.date_range(start='2020/7/1', end='2023/6/30', freq='ME')
sales = pd.read_excel(path_file)
# 创建DataFrame
df = sales

# 绘制原始数据
plt.figure(figsize=(12,6))
plt.plot(df.index, df[target_column], label='全销量')
plt.title('月度销量数据')
plt.xlabel('日期')
plt.ylabel('销量')
plt.show()

# ADF测试
result = adfuller(df[target_column])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# 差分
df['Sales_diff'] = df[target_column].diff()
df['Sales_diff'].dropna(inplace=True)

# 绘制ACF和PACF图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8))
plot_acf(df['Sales_diff'], ax=ax1, lags=40)
plot_pacf(df['Sales_diff'], ax=ax2, lags=40)
plt.show()

# 拆分训练集和测试集
train = df[target_column]

# 创建和训练ARIMA模型
model = ARIMA(train, order=(1,1,1))
results = model.fit()
# 预测
forecast = results.forecast(steps=7)


# 创建预测日期范围
last_date = train.index[-1]
forecast_dates = pd.date_range(start=last_date , periods=len(forecast))
# 绘制预测结果
plt.figure(figsize=(12,6))
plt.plot(train.index, train, label='训练数据')
plt.plot(forecast_dates, forecast, label='预测销量')
plt.title('ARIMA模型销量预测')
plt.xlabel('日期')
plt.ylabel('销量')
plt.legend()
plt.show()

datas_1 = np.arange(1,8)
plt.figure(figsize=(12,6))
plt.plot( datas_1, forecast, label='预测销量')
plt.title('ARIMA模型销量预测')
plt.xlabel('日期')
plt.ylabel('销量')
plt.legend()
plt.show()
