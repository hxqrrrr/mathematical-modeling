import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def decision_tree_model(df_start):
    # 准备特征和标签
    features = df_start[['二氧化硅(SiO2)', '氧化钠(Na2O)', '氧化钾(K2O)', '氧化钙(CaO)',
                         '氧化镁(MgO)', '氧化铝(Al2O3)', '氧化铁(Fe2O3)', '氧化铜(CuO)',
                         '氧化铅(PbO)', '氧化钡(BaO)', '五氧化二磷(P2O5)', '氧化锶(SrO)',
                         '氧化锡(SnO2)', '二氧化硫(SO2)']]
    labels = df_start['类型']

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 训练决策树模型
    clf = DecisionTreeClassifier(random_state=42, max_depth=100, min_samples_split=2)
    clf.fit(X_train, y_train)

    # 评估模型
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return clf, features.columns

# 读取数据并进行预处理
df = pd.read_excel('合并总表.xlsx', sheet_name='Sheet1')
df = df.drop(df[(df['指数'] < 85) | (df['指数'] > 105)].index)
df_weathered = df.drop(df[df['表面风化'] == "无风化"].index)
df_unweathered = df.drop(df[df['表面风化'] == "风化"].index)
# 训练模型

model, feature_columns = decision_tree_model(df_unweathered)

# 定义一个函数来预测新数据
def predict_new_sample(model, feature_columns, new_sample):
    # 确保新样本具有与训练数据相同的列
    new_sample = new_sample[feature_columns]

    # 预测
    prediction = model.predict(new_sample)
    return prediction[0]

# 示例：如何使用这个函数来预测新的样本
new_sample = pd.DataFrame({
    '二氧化硅(SiO2)': [78.45], '氧化钠(Na2O)': [0], '氧化钾(K2O)': [0], '氧化钙(CaO)': [6.08],
    '氧化镁(MgO)': [1.86], '氧化铝(Al2O3)': [7.23], '氧化铁(Fe2O3)': [2.15], '氧化铜(CuO)': [2.11],
    '氧化铅(PbO)': [0], '氧化钡(BaO)': [0], '五氧化二磷(P2O5)': [1.06], '氧化锶(SrO)': [0.03],
    '氧化锡(SnO2)': [0.00], '二氧化硫(SO2)': [0.51],
})

predicted_class = predict_new_sample(model, feature_columns, new_sample)
print(f"预测的类型是: {predicted_class}")
