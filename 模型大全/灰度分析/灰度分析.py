import numpy as np
import matplotlib.pyplot as plt


def gm11(x0, n):
    x1 = np.cumsum(x0)
    z1 = (x1[:-1] + x1[1:]) / 2.0

    B = np.vstack([-z1, np.ones(len(x0) - 1)]).T
    Y = x0[1:]

    # 修改这一行
    params = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y)
    a, b = params[0], params[1]  # 直接获取 a 和 b

    # 预测部分保持不变
    x1_pred = [(x0[0] - b / a) * np.exp(-a * (k + 1)) + b / a for k in range(len(x0) + n)]
    x0_pred = np.hstack([x0[0], np.diff(x1_pred)])

    return x0_pred


# 示例数据
x0 = np.array([1.2, 2.1, 2.8, 3.6, 4.5, 5.7])

# 预测未来3个点
x0_pred = gm11(x0, 3)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(x0) + 1), x0, 'ro-', label='Original Data')
plt.plot(range(1, len(x0_pred) + 1), x0_pred, 'b*-', label='Predicted Data')
plt.axvline(x=len(x0), color='g', linestyle='--', label='Prediction Start')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('GM(1,1) Model Prediction')
plt.legend()
plt.grid(True)
plt.show()
