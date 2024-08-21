import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置中文字体，确保中文显示正确
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def calculate_average_interval(row):
    # 找出所有供货的日期（非零值的索引）
    supply_days = np.where(row != 0)[0]

    if len(supply_days) == 0:
        return np.nan  # 如果没有供货，返回NaN
    elif len(supply_days) == 1:
        return row.shape[0] - 1  # 如果只有一次供货，返回总天数减1
    else:
        # 计算相邻供货日期之间的间隔
        intervals = np.diff(supply_days)

        # 计算平均间隔
        avg_interval = np.mean(intervals)

        # 考虑首次供货前和最后一次供货后的间隔
        first_interval = supply_days[0]
        last_interval = row.shape[0] - 1 - supply_days[-1]

        # 将首尾间隔纳入计算
        total_intervals = np.sum(intervals) + first_interval + last_interval
        total_count = len(intervals) + 2

        return total_intervals / total_count
def calculate_average_consecutive_days(row):
    # 将行转换为二进制序列，1表示有供货，0表示无供货
    binary_sequence = (row != 0).astype(int)

    # 使用diff找出连续序列的开始和结束
    diff_sequence = np.diff(binary_sequence)
    start_indices = np.where(diff_sequence == 1)[0] + 1
    end_indices = np.where(diff_sequence == -1)[0] + 1

    # 处理边界情况
    if binary_sequence[0] == 1:
        start_indices = np.insert(start_indices, 0, 0)
    if binary_sequence[-1] == 1:
        end_indices = np.append(end_indices, len(binary_sequence))

    # 计算每个连续供货期的长度
    consecutive_lengths = end_indices - start_indices

    # 如果没有连续供货期，返回0
    if len(consecutive_lengths) == 0:
        return 0

    # 返回平均连续天数
    return np.mean(consecutive_lengths)
def min_max_normalize_profit(column):
    return (column - column.min()) / (column.max() - column.min())
def min_max_normalize_cost(column):
    return (column.max()-column) / (column.max() - column.min())


def entropy_weight_method(data):
    """
    计算给定数据的熵权并返回权重和综合得分。

    参数:
    data (pd.DataFrame): 输入数据，每行代表一个样本，每列代表一个指标。

    返回:
    tuple: (权重, 综合得分)
    """
    # 数据归一化
    data_normalized = data.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # 计算熵值
    n, m = data_normalized.shape
    k = 1 / np.log(n)

    # 计算第j个指标下，第i个样本占该指标的比重
    p = data_normalized / data_normalized.sum()

    # 计算熵值
    e = -k * (p * np.log(p + 1e-10)).sum()  # 加上一个小数避免log(0)

    # 计算权重
    w = (1 - e) / (1 - e).sum()

    # 计算综合得分
    scores = (data_normalized * w).sum(axis=1)

    return w, scores


def topsis(decision_matrix, weights, criteria, threshold=None):
    """
    改进的TOPSIS方法

    参数:
    decision_matrix : numpy array
        决策矩阵
    weights : numpy array
        权重
    criteria : list
        准则类型（1表示效益型，-1表示成本型）
    threshold : float, optional
        最低可接受得分阈值

    返回:
    pandas DataFrame
        包含原始得分、相对得分和排名
    """

    # 最大最小值标准化
    min_vals = np.min(decision_matrix, axis=0)
    max_vals = np.max(decision_matrix, axis=0)
    normalized_matrix = (decision_matrix - min_vals) / (max_vals - min_vals)

    # 构造加权标准化决策矩阵
    weighted_normalized_matrix = normalized_matrix * weights

    # 确定理想解和负理想解
    ideal_best = np.max(weighted_normalized_matrix, axis=0) * criteria
    ideal_worst = np.min(weighted_normalized_matrix, axis=0) * criteria

    # 计算距离
    s_best = np.sqrt(np.sum((weighted_normalized_matrix - ideal_best) ** 2, axis=1))
    s_worst = np.sqrt(np.sum((weighted_normalized_matrix - ideal_worst) ** 2, axis=1))

    # 计算相对接近度
    c_i = s_worst / (s_best + s_worst)

    # 计算相对得分（最高分为100）
    relative_scores = c_i / np.max(c_i) * 100

    # 创建结果DataFrame
    result = pd.DataFrame({
        'Original Score': c_i,
        'Relative Score': relative_scores,
    })

    # 正确计算排名（得分高的排名靠前）
    result['Rank'] = result['Original Score'].rank(method='min', ascending=False).astype(int)

    # 应用阈值（如果提供）
    if threshold is not None:
        result = result[result['Relative Score'] >= threshold]

    return result.sort_values('Rank')
# 读取数据
file_path = '附件1 近5年402家供应商的相关数据.xlsx'
if os.path.exists(file_path):
    data_order = pd.read_excel(file_path, sheet_name=0)
    data_supply=pd.read_excel(file_path, sheet_name=1)
    print("数据已成功读取。")

# data_order.head()
# data_supply.head()
# 数据清洗和特征提取
data_order_feature = pd.DataFrame({
    '供应商':data_order.iloc[:,0],
    '材料分类':data_order['材料分类'],
    'sum':data_supply.iloc[:,2:].sum(axis=1),
    'max':data_supply.iloc[:,2:].max(axis=1),
    '供货次数':(data_supply.iloc[:,2:] > 0).sum(axis=1),
    'MSE':((data_supply.iloc[:,2:]-data_order.iloc[:,2:])**2).mean(axis=1),
    '间隔周数':(data_supply.iloc[:,2:]==0).sum(axis=1),
    '平均间隔周数':data_supply.iloc[:, 2:].apply(calculate_average_interval, axis=1),
    '平均连续天数':data_supply.iloc[:, 2:].apply(calculate_average_consecutive_days, axis=1),
    '合理性':((data_supply.iloc[:, 2:] / data_order.iloc[:, 2:].replace(0, np.nan) < 1.2) &
     (data_supply.iloc[:, 2:] / data_order.iloc[:, 2:].replace(0, np.nan) > 0.8)).sum(axis=1) / (data_order.iloc[:, 2:] != 0).sum(axis=1),
    '间隔周数(归一化)':min_max_normalize_cost((data_supply.iloc[:,2:]==0).sum(axis=1)),
    '平均间隔周数(归一化)':min_max_normalize_profit(data_supply.iloc[:, 2:].apply(calculate_average_interval, axis=1)),
    '平均连续天数(归一化)':min_max_normalize_profit(data_supply.iloc[:, 2:].apply(calculate_average_consecutive_days, axis=1)),

})

# 保存处理后的数据
# data_order_feature.to_csv("重要性模型的数据处理.csv", index=False)
print("数据处理完成。")

df_stablize = data_order_feature[['间隔周数(归一化)', '平均间隔周数(归一化)', '平均连续天数(归一化)']]

weights, scores = entropy_weight_method(df_stablize)
df_stablize['综合得分'] = scores
data_order_feature['稳定性(熵权法）'] = scores

data_order_feature_normalized =pd.DataFrame({
    '供应商':data_order.iloc[:,0],
    'sum':min_max_normalize_profit(data_supply.iloc[:,2:].sum(axis=1)),
    'max':min_max_normalize_profit(data_supply.iloc[:,2:].max(axis=1)),
    '供货次数':min_max_normalize_profit((data_supply.iloc[:,2:] > 0).sum(axis=1)),
    'MSE':min_max_normalize_cost(((data_supply.iloc[:,2:]-data_order.iloc[:,2:])**2).mean(axis=1)),
    '合理性':min_max_normalize_profit(((data_supply.iloc[:, 2:] / data_order.iloc[:, 2:].replace(0, np.nan) < 1.2) &
     (data_supply.iloc[:, 2:] / data_order.iloc[:, 2:].replace(0, np.nan) > 0.8)).sum(axis=1) / (data_order.iloc[:, 2:] != 0).sum(axis=1)),
    '稳定性(熵权法）':min_max_normalize_profit(scores),
})
df_importance = data_order_feature_normalized[['sum', 'max','供货次数', 'MSE', '合理性', '稳定性(熵权法）']]
importance_weights, importance_scores = entropy_weight_method(df_importance)
data_order_feature_normalized['重要性得分'] = importance_scores

# //topsis方法
decision_matrix = data_order_feature_normalized[['sum', 'max','供货次数', 'MSE', '合理性', '稳定性(熵权法）']].values
weights = np.array(importance_weights)
criteria = [1, 1, 1, -1, 1, 1]
topsis_result = topsis(decision_matrix, weights, criteria)
topsis_result['供应商'] = data_order_feature_normalized['供应商']
result = topsis_result[topsis_result['Rank'] <= 50]
result['材料分类'] = data_order_feature['材料分类']
topsis_result['材料分类'] = data_order_feature['材料分类']
print(result)
topsis_result.to_csv('重要性模型的TOPSIS结果.csv')
result.to_csv("重要性模型的前50名供应商结果.csv", index=False)

