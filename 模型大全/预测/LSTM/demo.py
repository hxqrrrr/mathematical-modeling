import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# 获取苹果公司的股票数据
ticker = "AAPL"
start_date = "2010-01-01"
end_date = "2023-08-15"

data = yf.download(ticker, start=start_date, end=end_date)

# 使用收盘价
close_prices = data['Close'].values.reshape(-1, 1)

# 数据标准化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# 准备数据
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60  # 使用过去60天的数据来预测
X, Y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 分割训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# 构建LSTM模型
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=0)

# 使用模型进行预测
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 反向转换数据
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

# 创建两个子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# 绘制原始股价数据
ax1.plot(data.index, close_prices, label='Original Stock Price')
ax1.set_title(f'{ticker} Stock Price Data')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')
ax1.legend()

# 绘制预测结果与实际价格对比
ax2.plot(data.index[look_back:train_size+look_back], Y_train[0], label='Actual Price (Train)')
ax2.plot(data.index[look_back:train_size+look_back], train_predict[:,0], label='Predicted Price (Train)')
ax2.plot(data.index[train_size+look_back:], Y_test[0], label='Actual Price (Test)')
ax2.plot(data.index[train_size+look_back:], test_predict[:,0], label='Predicted Price (Test)')
ax2.set_title(f'{ticker} Stock Price Prediction')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
ax2.legend()

plt.tight_layout()
plt.show()

# 计算并打印均方误差
train_mse = np.mean((Y_train[0] - train_predict[:,0])**2)
test_mse = np.mean((Y_test[0] - test_predict[:,0])**2)
print(f"Train Mean Squared Error: {train_mse}")
print(f"Test Mean Squared Error: {test_mse}")
