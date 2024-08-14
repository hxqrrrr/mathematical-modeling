import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.animation import FuncAnimation


class MM1Queue:
    def __init__(self, arrival_rate, service_rate):
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.queue = deque()
        self.current_time = 0
        self.next_arrival = self.generate_interarrival()
        self.next_departure = float('inf')
        self.total_wait = 0
        self.customers_served = 0
        self.max_queue_length = 0
        self.queue_length_history = []
        self.time_history = []

    def generate_interarrival(self):
        return np.random.exponential(1 / self.arrival_rate)

    def generate_service_time(self):
        return np.random.exponential(1 / self.service_rate)

    def run(self, simulation_time):
        while self.current_time < simulation_time:
            if self.next_arrival < self.next_departure:
                self.current_time = self.next_arrival
                self.handle_arrival()
            else:
                self.current_time = self.next_departure
                self.handle_departure()

            self.queue_length_history.append(len(self.queue))
            self.time_history.append(self.current_time)

    def handle_arrival(self):
        self.queue.append(self.current_time)
        self.next_arrival = self.current_time + self.generate_interarrival()
        if len(self.queue) == 1 and self.next_departure == float('inf'):
            self.next_departure = self.current_time + self.generate_service_time()
        self.max_queue_length = max(self.max_queue_length, len(self.queue))

    def handle_departure(self):
        arrival_time = self.queue.popleft()
        self.total_wait += self.current_time - arrival_time
        self.customers_served += 1
        if self.queue:
            self.next_departure = self.current_time + self.generate_service_time()
        else:
            self.next_departure = float('inf')


def animate(i):
    ax.clear()
    ax.plot(queue.time_history[:i], queue.queue_length_history[:i])
    ax.set_xlim(0, simulation_time)
    ax.set_ylim(0, max(queue.queue_length_history) + 1)
    ax.set_title('Queue Length Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Queue Length')


# 设置参数
arrival_rate = 4  # 每小时平均到达5个顾客
service_rate = 6  # 每小时平均服务6个顾客
simulation_time = 100  # 模拟100小时

# 创建并运行模拟
queue = MM1Queue(arrival_rate, service_rate)
queue.run(simulation_time)

# 创建动画
fig, ax = plt.subplots(figsize=(10, 6))
ani = FuncAnimation(fig, animate, frames=len(queue.time_history), interval=50, repeat=False)

plt.show()

# 打印结果
print(f"模拟结果：")
print(f"Average Wait Time: {queue.total_wait / queue.customers_served}")
print(f"Customers Served: {queue.customers_served}")
print(f"Max Queue Length: {queue.max_queue_length}")

# 理论计算
rho = arrival_rate / service_rate
L = rho / (1 - rho)
W = 1 / (service_rate - arrival_rate)
Lq = L - rho
Wq = W - 1 / service_rate

print(f"\n理论结果：")
print(f"系统利用率 (ρ): {rho}")
print(f"平均系统内顾客数 (L): {L}")
print(f"平均等待时间 (W): {W}")
print(f"平均队列长度 (Lq): {Lq}")
print(f"平均队列等待时间 (Wq): {Wq}")
