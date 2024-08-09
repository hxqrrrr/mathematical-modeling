import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from math import sqrt
import itertools
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 设置参数
target_columns = ['水生根茎类','花叶类','食用菌','辣椒类','花菜类','茄类']
parent_dir = os.path.dirname(os.getcwd())
path_file = os.path.join(parent_dir, "解题代码", "分类全销量按日.xlsx")

dates = pd.date_range(start='2020/7/1', end='2023/6/30', freq='ME')
sales = pd.read_excel(path_file)
 # 创建DataFrame
df = sales
predict = pd.DataFrame()


# 网格搜索函数
def grid_search_arima(data, p_range, d_range, q_range):
    best_aic = float("inf")
    best_params = None
    best_model = None

    for p, d, q in itertools.product(p_range, d_range, q_range):
        try:
            model = ARIMA(data, order=(p, d, q))
            results = model.fit()
            aic = results.aic
            if aic < best_aic:
                best_aic = aic
                best_params = (p, d, q)
                best_model = results
            print(f'ARIMA({p},{d},{q}) AIC: {aic}')
        except:
            continue

    return best_model, best_params, best_aic

def item_forecast(target_column):




    # ADF测试
    result = adfuller(df[target_column])
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])

    # 差分
    df['Sales_diff'] = df[target_column].diff()
    df['Sales_diff'].dropna(inplace=True)

    # 拆分训练集和测试集
    train = df[target_column][-100:]
    # 定义参数范围
    p_range = range(0, 3)
    d_range = range(0, 3)
    q_range = range(0, 3)

    # 执行网格搜索
    best_model, best_params, best_aic = grid_search_arima(train, p_range, d_range, q_range)

    print(f"\nBest ARIMA{best_params} model - AIC:{best_aic}")

    # 使用最佳模型进行预测
    forecast = best_model.forecast(steps=7)
    predict[target_column] = forecast.reset_index(drop=True)
    # 打印预测结果
    print("\n预测结果:")
    print(forecast)
i = 0
while i<7:
    item_forecast(target_columns[i])
    i+=1
predict.to_excel('预测结果.xlsx')