import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置中文字体，确保中文显示正确
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 读取数据
file_path1 = '附件1 近5年402家供应商的相关数据.xlsx'
file_path2 = '38家核心供应商.csv'

data_supply = pd.read_excel(file_path1, sheet_name='供应商的供货量（m³）')
data_order = pd.read_excel(file_path1)
data_core = pd.read_csv(file_path2)

df = pd.DataFrame({
    '供应商ID': data_supply['供应商ID'],
    '材料分类':data_order['材料分类'],
    'sum_supply':data_supply.iloc[:,2:].sum(axis=1),
    'mean_supply':data_supply.iloc[:,2:].mean(axis=1),
    'sum_order':data_order.iloc[:,2:].sum(axis=1),
   'mean_order':data_order.iloc[:,2:].mean(axis=1),
    'core_supplier':data_core.iloc[:,0]
})

