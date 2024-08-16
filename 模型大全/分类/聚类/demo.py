import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

np.random.seed(123)

data = np.random.randn(10, 5)
df = pd.DataFrame(data, columns=[f'V{i+1}' for i in range(5)],
                  index=[f'S{i+1}' for i in range(10)])

def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)

r_clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
r_clustering.fit(df)

q_clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
q_clustering.fit(df.T)

plt.figure(figsize=(10, 5))
plot_dendrogram(r_clustering, labels=df.index, leaf_rotation=90)
plt.title('R型聚类结果')
plt.xlabel('样本')
plt.show()

plt.figure(figsize=(10, 5))
plot_dendrogram(q_clustering, labels=df.columns, leaf_rotation=90)
plt.title('Q型聚类结果')
plt.xlabel('变量')
plt.show()

linkage_r = linkage(df, method='ward')
linkage_q = linkage(df.T, method='ward')

plt.figure(figsize=(10, 8))
sns.clustermap(df, row_linkage=linkage_r, col_linkage=linkage_q,
               cmap='viridis', figsize=(10, 8))
plt.title('R型和Q型聚类结合的热图')
plt.show()
