import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置中文字体，确保中文显示正确
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 读取数据
file_path = '附件1 近5年402家供应商的相关数据.xlsx'

data = pd.read_excel(file_path)
df = pd.DataFrame({
    '供应商ID': data['供应商ID'],
    'sum':data.iloc[:,2:].sum(axis=1),
    'mean':data.iloc[:,2:].mean(axis=1),

})


