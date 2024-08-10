import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

# 获取当前脚本所在目录的绝对路径
current_path = os.path.dirname(__file__)
path_file = os.path.join(current_path, '合并总表.xlsx')
df = pd.read_excel(path_file)

features = [
    '二氧化硅(SiO2)', '氧化钠(Na2O)', '氧化钾(K2O)', '氧化钙(CaO)', '氧化镁(MgO)',
    '氧化铝(Al2O3)', '氧化铁(Fe2O3)', '氧化铜(CuO)', '氧化铅(PbO)', '氧化钡(BaO)',
    '五氧化二磷(P2O5)', '氧化锶(SrO)', '氧化锡(SnO2)', '二氧化硫(SO2)'
]

X = df[features].values

# 3. 数据标准化
print("正在标准化数据...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 进行K-means聚类
print("正在进行K-means聚类...")
n_clusters = 4  # 您可以根据需要调整这个数字
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# 将聚类结果添加到原始数据框
df['聚类'] = cluster_labels + 1  # 加1是为了让聚类标签从1开始而不是0

# 5. 使用PCA进行降维和可视化
print("正在进行PCA降维和可视化...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
plt.colorbar(scatter)
plt.title('K-means聚类结果可视化 (PCA)')
plt.xlabel('第一主成分')
plt.ylabel('第二主成分')
plt.savefig('kmeans_cluster_visualization.png')
plt.close()

# 6. 分析每个聚类的特征
print("正在分析聚类特征...")
cluster_means = df.groupby('聚类')[features].mean()
print("各聚类的平均化学成分：")
print(cluster_means)

# 7. 保存结果
print("正在保存结果...")
df.to_excel('kmeans_clustering_results.xlsx', index=False)
cluster_means.to_excel('kmeans_cluster_means.xlsx')

print("分析完成！结果已保存。")

# 8. 输出一些基本统计信息
print("\n基本统计信息：")
print(f"总样本数：{len(df)}")
print("各聚类的样本数：")
print(df['聚类'].value_counts().sort_index())

# 9. 找出每个聚类的代表性样本
print("\n每个聚类的代表性样本：")
for cluster in range(1, n_clusters + 1):
    cluster_samples = df[df['聚类'] == cluster]
    cluster_center = cluster_means.loc[cluster]
    distances = ((cluster_samples[features] - cluster_center) ** 2).sum(axis=1)
    representative = cluster_samples.loc[distances.idxmin(), '文物编号']
    print(f"聚类 {cluster} 的代表性样本：{representative}")
