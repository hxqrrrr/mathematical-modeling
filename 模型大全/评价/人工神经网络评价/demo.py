import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 1. 准备数据
# 这里我们使用模拟数据，实际应用中您应该使用真实的房屋数据
np.random.seed(0)
n_samples = 1000

area = np.random.randint(50, 300, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
age = np.random.randint(0, 50, n_samples)

# 创建一个简单的价格模型（实际情况会更复杂）
price = (area * 1000 + bedrooms * 50000 + bathrooms * 30000 - age * 1000
         + np.random.normal(0, 50000, n_samples))

# 创建DataFrame
df = pd.DataFrame({
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age': age,
    'price': price
})

# 2. 数据预处理
X = df[['area', 'bedrooms', 'bathrooms', 'age']]
y = df['price']

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 构建神经网络模型
model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 4. 训练模型
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32,
                    validation_split=0.2, verbose=0)

# 5. 评估模型
train_loss = model.evaluate(X_train_scaled, y_train, verbose=0)
test_loss = model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"训练集 MSE: {train_loss:.2f}")
print(f"测试集 MSE: {test_loss:.2f}")

# 6. 使用模型进行预测
sample_house = np.array([[150, 3, 2, 10]])  # 面积150平方米，3卧室，2浴室，10年房龄
sample_house_scaled = scaler.transform(sample_house)
predicted_price = model.predict(sample_house_scaled)

print(f"预测价格: ${predicted_price[0][0]:,.2f}")

# 7. 可视化训练过程
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型训练过程')
plt.xlabel('Epoch')
plt.ylabel('均方误差损失')
plt.legend()
plt.show()
