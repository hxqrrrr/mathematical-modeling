import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import stats
from pygam import GAM, s
from sklearn.model_selection import train_test_split
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 读取数据
current_path = os.path.dirname(__file__)
path_file1 = os.path.join(current_path, '原始数据_示例数据青少年体质数据.xlsx')
df = pd.read_excel(path_file1)

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df['年龄'].values.reshape(-1, 1), df['身高cm'], test_size=0.2, random_state=42)

# 拟合GAM模型
gam = GAM(s(0, n_splines=10))
gam.fit(X_train, y_train)

# 计算R²值
r_squared_gam = gam.score(X_test, y_test)

# 生成预测值
X_pred = np.linspace(df['年龄'].min(), df['年龄'].max(), 100).reshape(-1, 1)
y_pred = gam.predict(X_pred)

# 计算置信区间
ci = gam.confidence_intervals(X_pred, width=0.95)

# 打印ci的形状和内容，以了解它的结构
print("CI shape:", ci.shape)
print("CI content:", ci)

# 绘图
plt.figure(figsize=(12, 8))
plt.scatter(df['年龄'], df['身高cm'], alpha=0.5, label='原始数据')
plt.plot(X_pred, y_pred, 'r-', label='GAM拟合')

# 根据ci的实际结构来绘制置信区间
if ci.shape[1] == 2:
    plt.fill_between(X_pred.ravel(), ci[:, 0], ci[:, 1], alpha=0.3, label='95% 置信区间')
else:
    print("置信区间的结构与预期不符，请检查ci的内容")

plt.title('青少年年龄与身高关系图 (GAM拟合)')
plt.xlabel('年龄')
plt.ylabel('身高 (cm)')
plt.legend()
plt.show()

print(f"GAM模型的R²值: {r_squared_gam:.4f}")
