from pulp import *

# 创建一个最大化问题
prob = LpProblem("整数规划示例", LpMaximize)

# 定义变量
x = LpVariable("x", lowBound=0, cat='Integer')
y = LpVariable("y", lowBound=0, cat='Integer')

# 设置目标函数
prob += 3*x + 2*y, "利润"

# 添加约束条件
prob += 2*x + y <= 100, "材料A约束"
prob += x + y <= 80, "材料B约束"
prob += x <= 40, "产品X的需求约束"

# 求解问题
prob.solve()

# 输出结果
print("状态:", LpStatus[prob.status])
print("最优解:")
print("x =", value(x))
print("y =", value(y))
print("最大利润 =", value(prob.objective))
