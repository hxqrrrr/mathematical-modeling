import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置中文字体，确保中文显示正确
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 读取数据
file_path = '附件1 近5年402家供应商的相关数据.xlsx'
if os.path.exists(file_path):
    data_order = pd.read_excel(file_path, sheet_name=0)
    data_supply=pd.read_excel(file_path, sheet_name=1)
    print("数据已成功读取。")

# data_order.head()
# data_supply.head()

# 数据清洗
