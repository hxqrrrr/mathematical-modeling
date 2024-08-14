import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# 总人口
N = 10000
# 初始感染人数
I0, R0 = 100, 0
S0 = N - I0 - R0
# 感染率和恢复率
beta, gamma = 0.3, 0.1

# 初始条件
y0 = S0, I0, R0

# 时间点
t = np.linspace(0, 100, 100)

# 求解 ODE
solution = odeint(sir_model, y0, t, args=(N, beta, gamma))
S, I, R = solution.T

# 绘图
plt.figure(figsize=(10,6))
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of people')
plt.title('SIR Model')
plt.legend()
plt.grid(True)
plt.show()
