import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

# 1. 加载数据
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# 2. 数据预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 创建和训练SVM模型
svm_classifier = svm.SVC(kernel='rbf', random_state=42)
svm_classifier.fit(X_train_scaled, y_train)

# 4. 模型评估
y_pred = svm_classifier.predict(X_test_scaled)
print("初始模型性能：")
print(classification_report(y_test, y_pred))

# 5. 参数调优
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

grid_search = GridSearchCV(svm.SVC(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

print("最佳参数：", grid_search.best_params_)
best_model = grid_search.best_estimator_

# 6. 使用最佳模型进行预测
y_pred_best = best_model.predict(X_test_scaled)
print("\n最佳模型性能：")
print(classification_report(y_test, y_pred_best))

# 7. 混淆矩阵
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('混淆矩阵')
plt.colorbar()
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.show()

# 8. 可视化决策边界（使用PCA降维到2D）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=42)

best_model_2d = svm.SVC(**grid_search.best_params_)
best_model_2d.fit(X_train_pca, y_train_pca)


def plot_decision_boundary(X, y, model, ax=None):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    if ax is None:
        ax = plt.gca()
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
    ax.set_xlabel('第一主成分')
    ax.set_ylabel('第二主成分')
    return ax


plt.figure(figsize=(12, 5))
ax1 = plt.subplot(121)
plot_decision_boundary(X_train_pca, y_train_pca, best_model_2d, ax=ax1)
ax1.set_title('训练集')

ax2 = plt.subplot(122)
plot_decision_boundary(X_test_pca, y_test_pca, best_model_2d, ax=ax2)
ax2.set_title('测试集')

plt.tight_layout()
plt.show()

# 9. 特征重要性（基于SVM系数，仅适用于线性核）
if best_model.kernel == 'linear':
    feature_importance = abs(best_model.coef_[0])
    feature_names = breast_cancer.feature_names
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    plt.figure(figsize=(12, 6))
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, np.array(feature_names)[sorted_idx])
    plt.xlabel('特征重要性')
    plt.title('SVM特征重要性（仅适用于线性核）')
    plt.tight_layout()
    plt.show()
else:
    print("注意：特征重要性仅适用于线性核SVM。当前最佳模型使用的是非线性核。")
