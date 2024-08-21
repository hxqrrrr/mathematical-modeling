import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
def decision_tree_model(df_start):
    # 准备特征和标签
    df_encoded = pd.get_dummies(df_start, columns=['纹饰', '颜色'])
    features = df_encoded[['二氧化硅(SiO2)', '氧化钠(Na2O)', '氧化钾(K2O)', '氧化钙(CaO)',
                           '氧化镁(MgO)', '氧化铝(Al2O3)', '氧化铁(Fe2O3)', '氧化铜(CuO)',
                           '氧化铅(PbO)', '氧化钡(BaO)', '五氧化二磷(P2O5)', '氧化锶(SrO)',
                           '氧化锡(SnO2)', '二氧化硫(SO2)'] + list(df_encoded.columns[df_encoded.columns.str.startswith(('纹饰_', '颜色_'))])]
    labels = df_start['类型']

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 训练决策树模型
    clf = DecisionTreeClassifier(random_state=42,max_depth=100,min_samples_split=2)  # 移除 max_depth 限制以查看完整树
    clf.fit(X_train, y_train)

    # 可视化完整决策树
    plt.figure(figsize=(40, 20))  # 增大图像尺寸以适应完整树
    plot_tree(clf, feature_names=features.columns, class_names=clf.classes_, filled=True, fontsize=10)
    plt.title('完整决策树可视化')
    plt.show()

    # 评估模型
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
# 假设current_path已经定义，文件路径已经设置
# 读取数据并进行预处理
df = pd.read_excel('合并总表.xlsx', sheet_name='Sheet1')
df = df.drop(df[(df['指数'] < 85) | (df['指数'] > 105)].index)
df_weathered = df.drop(df[df['表面风化'] == "无风化"].index)
df_unweathered = df.drop(df[df['表面风化'] == "风化"].index)

df_start = df_weathered
# df_start = df_unweathered
#分化或者未风化，用注释修改变量
decision_tree_model(df_start)