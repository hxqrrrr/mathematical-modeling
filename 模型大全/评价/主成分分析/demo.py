import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取数据
df = pd.read_excel('原始数据_示例数据青少年体质数据.xlsx')

# 选择数值型变量
numeric_columns = ['年龄', '身高cm', '体重kg', '肺活量', '舒张压', '收缩压', '心率', '最大心率', '最大吸氧量', '负荷时间', '做功']

# 标准化数据
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numeric_columns])

# 进行PCA
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

# 获取载荷
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=numeric_columns
)

# 打印载荷
print("主成分载荷矩阵:")
print(loadings)

# 打印解释方差比
print("\n各主成分解释方差比:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")

# 可视化前两个主成分的载荷
plt.figure(figsize=(10, 6))
for i, var in enumerate(numeric_columns):
    plt.arrow(0, 0, loadings.iloc[i, 0], loadings.iloc[i, 1], head_width=0.05, head_length=0.05)
    plt.text(loadings.iloc[i, 0], loadings.iloc[i, 1], var)

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA载荷图')
plt.grid()
plt.show()
