import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
class ImprovedBPNN:
    def __init__(self, layers):
        self.layers = layers
        self.weights = self.he_init(layers)
        self.biases = [np.zeros((1, l)) for l in layers[1:]]
        self.learning_rate = 0.01
        self.l2_lambda = 0.01

    def he_init(self, layers):
        return [np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2 / layers[i]) for i in range(len(layers) - 1)]

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.relu(z) if i != len(self.weights) - 1 else z  # 输出层不使用激活函数
            self.activations.append(a)
        return self.activations[-1]

    def backward(self, X, y):
        m = X.shape[0]
        delta = self.activations[-1] - y
        deltas = [delta]

        for i in range(len(self.layers) - 2, 0, -1):
            delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z_values[i - 1])
            deltas.append(delta)

        deltas.reverse()

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * (
                        np.dot(self.activations[i].T, deltas[i]) / m + self.l2_lambda * self.weights[i])
            self.biases[i] -= self.learning_rate * np.mean(deltas[i], axis=0)

    def train(self, X, y, epochs, batch_size=32):
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                self.forward(X_batch)
                self.backward(X_batch, y_batch)

            if epoch % 100 == 0:
                loss = np.mean((self.predict(X) - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        return self.forward(X)


# 使用示例
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X) + np.random.normal(0, 0.1, X.shape)

# 标准化数据
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

nn = ImprovedBPNN([1, 20, 20, 1])
nn.train(X_scaled, y_scaled, epochs=1000, batch_size=32)

# 预测
predictions_scaled = nn.predict(X_scaled)
predictions = scaler_y.inverse_transform(predictions_scaled)

# 可视化结果
plt.scatter(X, y, label='实际数据')
plt.plot(X, predictions, color='red', label='预测')
plt.legend()
plt.title('改进后的BP神经网络预测')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
