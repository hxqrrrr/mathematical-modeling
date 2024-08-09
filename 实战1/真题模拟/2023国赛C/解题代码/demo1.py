import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy import stats
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 获取当前脚本所在目录的绝对路径
current_path = os.path.dirname(__file__)
# 拼接文件名
path_file4 = os.path.join(current_path, '附件4.xlsx')
df4 = pd.read_excel(path_file4)
df4

