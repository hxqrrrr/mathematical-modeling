import matplotlib.pyplot as plt
import numpy as np

# 数据
products = ['Product 1', 'Product 2']
quantities = [60.0, 0.0]
optimal_objective = 100

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 产品数量柱状图
ax1.bar(products, quantities, color=['#1f77b4', '#ff7f0e'])
ax1.set_ylabel('Quantity')
ax1.set_title('Optimal Product Quantities')
for i, v in enumerate(quantities):
    ax1.text(i, v, f'{v}', ha='center', va='bottom')

# 目标函数值饼图
sizes = [optimal_objective, 100 - optimal_objective]
labels = ['Objective Value', '']
colors = ['#2ca02c', '#d3d3d3']
ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax2.set_title('Optimal Objective Value')

# 添加一个文本框显示具体的目标函数值
ax2.text(0.5, -0.1, f'Optimal Objective: {optimal_objective}',
         horizontalalignment='center', verticalalignment='center',
         transform=ax2.transAxes, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
