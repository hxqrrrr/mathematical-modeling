import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 1. 加载数据
current_path = os.path.dirname(__file__)
path_file1 = os.path.join(current_path, '原始数据_示例数据青少年体质数据.xlsx')
df = pd.read_excel(path_file1)

# 2. 特征选择
features = ['年龄', '性别', '身高cm', '体重kg', '肺活量', '舒张压', '收缩压', '心率', '最大心率']
target = '最大吸氧量'

X = df[features]
y = df[target]

# 将性别转换为数值型
X['性别'] = X['性别'].map({1: 0, 2: 1})  # 假设1代表男性，2代表女性

# 3. 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. 构建BP神经网络模型
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\n模型结构：")
model.summary()

# 5. 训练模型
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1)

# 6. 评估模型
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\n测试集平均绝对误差: {test_mae:.4f}")

# 预测
predictions = model.predict(X_test)

# 7. 可视化结果
plt.figure(figsize=(12, 8))

# 训练过程中的损失变化
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失')
plt.xlabel('Epoch')
plt.ylabel('损失')
plt.legend()

# 训练过程中的MAE变化
plt.subplot(2, 2, 2)
plt.plot(history.history['mae'], label='训练MAE')
plt.plot(history.history['val_mae'], label='验证MAE')
plt.title('模型MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

# 预测值vs实际值的散点图
plt.subplot(2, 2, 3)
plt.scatter(y_test, predictions)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('预测值 vs 实际值')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

# 预测误差的直方图
plt.subplot(2, 2, 4)
errors = predictions.flatten() - y_test
plt.hist(errors, bins=30)
plt.xlabel('预测误差')
plt.ylabel('频数')
plt.title('预测误差分布')

plt.tight_layout()
plt.show()

# 8. 特征重要性分析
feature_importance = np.abs(model.layers[0].get_weights()[0]).mean(axis=1)
feature_importance = pd.Series(feature_importance, index=features)
feature_importance = feature_importance.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar')
plt.title('特征重要性')
plt.xlabel('特征')
plt.ylabel('重要性得分')
plt.tight_layout()
plt.show()

print("\n特征重要性排序：")
print(feature_importance)

# 9. 模型保存
model.save('youth_fitness_model.h5')
print("\n模型已保存为 'youth_fitness_model.h5'")
