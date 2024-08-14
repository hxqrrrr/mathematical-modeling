import numpy as np

# 判断矩阵
A = np.array([
    [1, 1/2, 1/3, 2],
    [2, 1, 1/2, 3],
    [3, 2, 1, 4],
    [1/2, 1/3, 1/4, 1]
])

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

# 找到最大特征值对应的特征向量
max_index = np.argmax(eigenvalues)
largest_eigenvector = eigenvectors[:, max_index].real

# 归一化特征向量得到权重
weights = largest_eigenvector / np.sum(largest_eigenvector)

print("权重向量:", weights)
print("最大特征值:", eigenvalues[max_index].real)

n = len(A)
RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}

CI = (eigenvalues[max_index].real - n) / (n - 1)
CR = CI / RI[n]

print("一致性比率 (CR):", CR)
