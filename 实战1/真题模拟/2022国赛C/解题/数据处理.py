import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 获取当前脚本所在目录的绝对路径
current_path = os.path.dirname(__file__)

# 构建附件1.xlsx和附件4.xlsx的相对路径
path_file1 = os.path.join(current_path, '表1.xlsx')
path_file2 = os.path.join(current_path, '表2.xlsx')
path_file = os.path.join(current_path, '合并总表.xlsx')
# 读取附件1.xlsx文件
df1 = pd.read_excel(path_file1, sheet_name='Sheet1')
df2 = pd.read_excel(path_file2, sheet_name='Sheet1')
df=pd.read_excel(path_file, sheet_name='Sheet1')
# #合并表格
# df = pd.merge(df1, df2, on='文物编号')
# df = df.fillna(0)
df = df.drop(df[(df['指数'] < 85) | (df['指数'] > 105)].index)
# df.to_excel("合并总表.xlsx")
