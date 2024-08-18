import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置中文字体，确保中文显示正确
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 读取数据
file_path1 = '重要性模型的数据处理.csv'
file_path2 = '附件1 近5年402家供应商的相关数据.xlsx'
data_order = pd.read_csv(file_path1)
data_附件1 = pd.read_excel(file_path2)

df = pd.DataFrame(data_order)
df = df[['供应商', 'sum']]
df['mean'] = df['sum'] / 240
df['材料分类'] = data_附件1['材料分类']
df['原料还原值'] = 0
# 材料倍数
df['原料还原值']=df['材料分类'].map({'A':1/0.6,'B':1/0.66,'C':1/0.72})*df['mean']
a = df['原料还原值'].sum()



