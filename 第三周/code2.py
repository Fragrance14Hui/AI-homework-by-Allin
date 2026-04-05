import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
def standardize(X):
    """数据标准化：(X - 均值) / 标准差"""
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    # 防止除以0
    X_std[X_std == 0] = 1e-8
    X_standardized = (X - X_mean) / X_std
    return X_standardized, X_mean, X_std

# ---------------------- 1. 数据加载与预处理 ----------------------
df = pd.read_csv("iris.data", header=None)
X = df.iloc[:, :-1].values  # 4个特征
y_true = df.iloc[:, -1].values  # 真实标签（用于评估）

# 标签编码（将字符串转为数字，仅用于评估）
label_map = {label: idx for idx, label in enumerate(np.unique(y_true))}
y_true_num = np.array([label_map[label] for label in y_true], dtype=int)  # 修正：指定int类型

# 数据标准化（修正点2：修复X_std重复赋值）
X_standardized, X_mean, X_std = standardize(X)  # 不再用X_std重复命名

# ---------------------- 2. K-Means算法实现 ----------------------
# 计算欧氏距离
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# K-Means主函数
def k_means(X, K, max_iters=100, tol=1e-6, random_state=42):
    np.random.seed(random_state)
    m, n = X.shape
    # 1. 初始化簇中心（随机选择K个样本）
    centroids_idx = np.random.choice(m, K, replace=False)
    centroids = X[centroids_idx].copy()
    # 存储每个样本的簇标签
    labels = np.zeros(m, dtype=int)  # 修正：指定int类型
    # 存储历史中心和每轮的标签（修正点3：记录每轮标签用于计算SSE）
    centroids_history = [centroids.copy()]
    labels_history = [labels.copy()]

    for iter in range(max_iters):
        # 2. 分配簇：计算每个样本到中心的距离，分配到最近的簇
        for i in range(m):
            distances = [euclidean_distance(X[i], centroid) for centroid in centroids]
            labels[i] = np.argmin(distances)
        
        # 3. 更新簇中心：计算每个簇的均值
        new_centroids = np.zeros((K, n))
        for k in range(K):
            cluster_samples = X[labels == k]
            if len(cluster_samples) == 0:
                # 空簇：随机重置中心
                new_centroids[k] = X[np.random.choice(m)]
            else:
                new_centroids[k] = np.mean(cluster_samples, axis=0)
        
        # 4. 检查收敛：中心变化小于阈值则停止
        centroid_change = np.sum([euclidean_distance(new_centroids[k], centroids[k]) for k in range(K)])
        centroids = new_centroids.copy()
        centroids_history.append(centroids.copy())
        labels_history.append(labels.copy())  # 记录当前轮标签
        
        if centroid_change < tol:
            print(f"K-Means收敛，迭代次数：{iter+1}")
            break
    
    return labels, centroids, centroids_history, labels_history  # 修正：返回标签历史

# 训练K-Means（K=3，对应真实类别数）
K = 3
# 修正：接收返回的labels_history
y_pred, centroids, centroids_hist, labels_hist = k_means(X_standardized, K=K, max_iters=100)

# ---------------------- 3. 聚类结果对齐（标签映射） ----------------------
# 因为K-Means的簇标签是随机的，需要将预测标签与真实标签对齐，才能计算准确率
def align_labels(y_true, y_pred, K):
    mapping = {}
    for k in range(K):
        # 找到预测簇k中，真实标签出现最多的类别
        true_labels_in_cluster = y_true[y_pred == k]
        if len(true_labels_in_cluster) == 0:
            mapping[k] = -1  # 空簇标记
            continue
        most_common = Counter(true_labels_in_cluster).most_common(1)[0][0]
        mapping[k] = most_common
    # 映射预测标签（处理空簇）
    y_pred_aligned = np.array([mapping[label] if mapping[label] != -1 else np.random.choice(y_true) for label in y_pred])
    return y_pred_aligned

# 对齐标签
y_pred_aligned = align_labels(y_true_num, y_pred, K)

# ---------------------- 4. 聚类模型评估 ----------------------
# 4.1 外部指标（有真实标签）：准确率、调整兰德指数(ARI)、归一化互信息(NMI)
def calculate_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# 调整兰德指数(ARI)
def adjusted_rand_index(y_true, y_pred):
    # 简化实现，核心逻辑：计算 contingency table
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    n = len(y_true)
    # 构建 contingency table
    contingency = np.zeros((len(labels_true), len(labels_pred)))
    for i, t in enumerate(labels_true):
        for j, p in enumerate(labels_pred):
            contingency[i, j] = np.sum((y_true == t) & (y_pred == p))
    # 计算ARI
    a = np.sum(contingency, axis=1)
    b = np.sum(contingency, axis=0)
    sum_comb_c = np.sum(contingency * (contingency - 1) / 2)
    sum_comb_a = np.sum(a * (a - 1) / 2)
    sum_comb_b = np.sum(b * (b - 1) / 2)
    expected = (sum_comb_a * sum_comb_b) / (n * (n - 1) / 2) if n > 1 else 0
    max_val = (sum_comb_a + sum_comb_b) / 2
    # 防止除零
    if max_val - expected == 0:
        return 0
    return (sum_comb_c - expected) / (max_val - expected)

# 归一化互信息(NMI)
def normalized_mutual_info(y_true, y_pred):
    # 计算互信息
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    n = len(y_true)
    # 概率分布
    p_true = np.array([np.sum(y_true == t) / n for t in labels_true])
    p_pred = np.array([np.sum(y_pred == p) / n for p in labels_pred])
    # 联合概率
    p_joint = np.zeros((len(labels_true), len(labels_pred)))
    for i, t in enumerate(labels_true):
        for j, p in enumerate(labels_pred):
            p_joint[i, j] = np.sum((y_true == t) & (y_pred == p)) / n
    # 互信息
    mi = 0
    for i in range(len(labels_true)):
        for j in range(len(labels_pred)):
            if p_joint[i, j] > 0:
                mi += p_joint[i, j] * np.log(p_joint[i, j] / (p_true[i] * p_pred[j]))
    # 熵
    h_true = -np.sum(p_true * np.log(p_true + 1e-10))  # 防止log(0)
    h_pred = -np.sum(p_pred * np.log(p_pred + 1e-10))
    # NMI（防止除零）
    if np.sqrt(h_true * h_pred) == 0:
        return 0
    return mi / np.sqrt(h_true * h_pred)

# 4.2 内部指标（无真实标签）：轮廓系数(Silhouette Score)、簇内平方和(SSE)
def silhouette_score(X, labels, K):
    m = X.shape[0]
    silhouette_vals = np.zeros(m)
    for i in range(m):
        # 样本i所在的簇
        cluster_i = labels[i]
        # 簇内平均距离a(i)
        same_cluster = X[labels == cluster_i]
        if len(same_cluster) == 1:
            silhouette_vals[i] = 0
            continue
        a_i = np.mean([euclidean_distance(X[i], x) for x in same_cluster if not np.array_equal(x, X[i])])
        # 最近的其他簇的平均距离b(i)
        b_i = np.inf
        for k in range(K):
            if k == cluster_i:
                continue
            other_cluster = X[labels == k]
            if len(other_cluster) == 0:
                continue
            dist = np.mean([euclidean_distance(X[i], x) for x in other_cluster])
            if dist < b_i:
                b_i = dist
        # 轮廓系数
        silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) != 0 else 0
    return np.mean(silhouette_vals)

# 簇内平方和(SSE)
def calculate_sse(X, labels, centroids, K):
    sse = 0
    for k in range(K):
        cluster_samples = X[labels == k]
        if len(cluster_samples) == 0:
            continue
        sse += np.sum([euclidean_distance(x, centroids[k]) ** 2 for x in cluster_samples])
    return sse

# 执行评估
print("\n" + "="*50)
print("K-Means聚类模型评估结果：")
acc = calculate_accuracy(y_true_num, y_pred_aligned)
ari = adjusted_rand_index(y_true_num, y_pred)
nmi = normalized_mutual_info(y_true_num, y_pred)
silhouette = silhouette_score(X_standardized, y_pred, K)
sse = calculate_sse(X_standardized, y_pred, centroids, K)

print(f"准确率(Accuracy): {acc:.4f}")
print(f"调整兰德指数(ARI): {ari:.4f}")
print(f"归一化互信息(NMI): {nmi:.4f}")
print(f"轮廓系数(Silhouette Score): {silhouette:.4f}")
print(f"簇内平方和(SSE): {sse:.4f}")

# ---------------------- 5. 可视化聚类结果 ----------------------
# 用PCA降维到2维（无sklearn，手动实现PCA）
def pca(X, n_components=2):
    # 1. 标准化（已标准化，直接计算协方差矩阵）
    cov_matrix = np.cov(X.T)
    # 2. 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # 处理复数特征值（修正点4：如果出现复数，取实部）
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    # 3. 按特征值降序排序
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    # 4. 投影到低维空间
    X_pca = X @ eigenvectors[:, :n_components]
    return X_pca

# 降维可视化
X_pca = pca(X_standardized, n_components=2)
centroids_pca = pca(centroids, n_components=2)

plt.figure(figsize=(10, 5))
# 聚类结果
plt.subplot(1, 2, 1)
for k in range(K):
    plt.scatter(X_pca[y_pred == k, 0], X_pca[y_pred == k, 1], label=f"Cluster {k}", alpha=0.7)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c="black", marker="X", s=200, label="Centroids")
plt.title("K-Means Clustering Result (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()

# 真实标签对比
plt.subplot(1, 2, 2)
for label in np.unique(y_true_num):
    plt.scatter(X_pca[y_true_num == label, 0], X_pca[y_true_num == label, 1], label=f"True Class {label}", alpha=0.7)
plt.title("True Iris Classes (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.show()

# 簇内平方和随迭代变化（修正点5：使用labels_history计算每轮SSE）
sse_history = []
for i, centroids_iter in enumerate(centroids_hist):
    # 取对应迭代的标签（如果超出长度，用最后一轮标签）
    labels_iter = labels_hist[i] if i < len(labels_hist) else y_pred
    sse_iter = calculate_sse(X_standardized, labels_iter, centroids_iter, K)
    sse_history.append(sse_iter)

plt.figure(figsize=(8, 4))
plt.plot(sse_history, marker="o")
plt.title("K-Means SSE vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Sum of Squared Errors (SSE)")
plt.grid(True)
plt.show()