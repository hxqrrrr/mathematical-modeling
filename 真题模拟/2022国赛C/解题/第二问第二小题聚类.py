import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取数据
df = pd.read_excel('合并总表.xlsx')
df = df.drop(df[(df['指数'] < 85) | (df['指数'] > 105)].index)
df_weathered = df.drop(df[df['表面风化'] == "无风化"].index)
df_unweathered = df.drop(df[df['表面风化'] == "风化"].index)
df_weathered_K = df_weathered[df_weathered['类型'] == "高钾"]
df_weathered_Ba = df_weathered[df_weathered['类型'] == "铅钡"]
df_unweathered_K = df_unweathered[df_unweathered['类型'] == "高钾"]
df_unweathered_Ba = df_unweathered[df_unweathered['类型'] == "铅钡"]
# 创建一个字典，将数据框与描述性名称关联起来
# 创建一个列表，每个元素是一个元组，包含DataFrame和对应的描述性名称
df_list = [
    (df_weathered_K, "风化高钾玻璃"),
    (df_weathered_Ba, "风化铅钡玻璃"),
    (df_unweathered_K, "无风化高钾玻璃"),
    (df_unweathered_Ba, "无风化铅钡玻璃")
]

# 遍历列表并保存
for df, name in df_list:
    filename = f"{name}_聚类结果.xlsx"
    df.to_excel(filename, index=False)
    print(f"已保存文件: {filename}")
