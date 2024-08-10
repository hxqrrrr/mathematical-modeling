import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 读取Excel文件（包含回归模型的参数）
model_data = pd.read_excel('kmeans_cluster_means.xlsx')

# 定义要分析的化学成分
compounds = ['二氧化硅(SiO2)', '氧化钠(Na2O)', '氧化钾(K2O)', '氧化钙(CaO)', '氧化镁(MgO)',
             '氧化铝(Al2O3)', '氧化铁(Fe2O3)', '氧化铜(CuO)', '氧化铅(PbO)', '氧化钡(BaO)',
             '五氧化二磷(P2O5)', '氧化锶(SrO)', '氧化锡(SnO2)', '二氧化硫(SO2)']


def clr_transform(df):
    """执行中心化对数比转换"""
    eps = 1e-10  # 小值，防止取对数时出现零
    log_data = np.log(df + eps)
    geometric_mean = log_data.mean(axis=1)
    return log_data.subtract(geometric_mean, axis=0)


def inverse_clr_transform(clr_values):
    """执行 CLR 转换的逆变换"""
    exp_values = np.exp(clr_values)
    sum_exp_values = np.sum(exp_values)
    return exp_values / sum_exp_values


def polynomial_regression(x, y, compound_name):
    # 标准化 x
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x.reshape(-1, 1))

    # 创建多项式特征
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(x_scaled)

    # 拟合多项式回归模型
    model = LinearRegression()
    model.fit(X_poly, y)

    # 计算R²值
    r2 = r2_score(y, model.predict(X_poly))

    return model.coef_, model.intercept_, r2, scaler.mean_[0], scaler.scale_[0]


def predict_compound(cluster_number, formula, mean, std):
    # 标准化聚类编号
    x_standardized = (cluster_number - mean) / std

    # 从公式中提取系数
    parts = formula.split('=')[1].strip().split()
    a = float(parts[0].split('x²')[0])
    b = float(parts[2].split('x')[0])
    c = float(parts[4])

    # 计算预测值
    prediction = a * x_standardized ** 2 + b * x_standardized + c

    return prediction


# 构建回归模型
formulas = []
x = model_data['聚类'].values
clr_data = clr_transform(model_data[compounds])

for compound in compounds:
    y = clr_data[compound].values
    coef, intercept, r2, mean, std = polynomial_regression(x, y, compound)
    formula = f"{compound} (CLR) = {coef[1]:.4f}x² + {coef[0]:.4f}x + {intercept:.4f}"
    formulas.append((compound, formula, r2, mean, std))

# 读取风化后的样品数据
weathered_samples = pd.read_excel('合并总表.xlsx')  # 请替换为您的实际文件名

# 对风化后的样品进行CLR转换
weathered_clr = clr_transform(weathered_samples[compounds])

# 定义风化程度范围（例如：0.1到1，其中1表示完全风化）
weathering_degrees = np.linspace(0.1, 1, 10)

results = []

for weathering_degree in weathering_degrees:
    # 预测聚类1的CLR值
    cluster_1_clr_predictions = []

    for compound, formula, r2, mean, std in formulas:
        cluster_1_prediction = predict_compound(1, formula, mean, std)

        # 考虑风化程度的影响
        weathered_value = weathered_clr[compound].mean()
        adjusted_prediction = (1 - weathering_degree) * cluster_1_prediction + weathering_degree * weathered_value

        cluster_1_clr_predictions.append(adjusted_prediction)

    # 将预测的CLR值转换回原始比例
    cluster_1_original_scale = inverse_clr_transform(cluster_1_clr_predictions)

    # 添加结果到列表
    results.append({
        'Weathering Degree': weathering_degree,
        **{compound: value for compound, value in zip(compounds, cluster_1_original_scale)}
    })

# 创建结果DataFrame
results_df = pd.DataFrame(results)

print("考虑不同风化程度的聚类1预测原始比例：")
print(results_df)

# 保存结果到Excel文件
results_df.to_excel('cluster_1_predictions_with_weathering.xlsx', index=False)
print("结果已保存到 'cluster_1_predictions_with_weathering.xlsx'")

# 可视化结果
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
for compound in compounds:
    plt.plot(results_df['Weathering Degree'], results_df[compound], label=compound)

plt.xlabel('Weathering Degree')
plt.ylabel('Predicted Original Proportion')
plt.title('Effect of Weathering on Predicted Original Proportions')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('weathering_effect_plot.png')
plt.show()
