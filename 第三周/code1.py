import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 解决可视化中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文黑体
plt.rcParams['axes.unicode_minus'] = False    # 负号显示

# 读取数据
df = pd.read_csv("winequality-red.csv", sep=";")

# 特征与标签分离
X = df.iloc[:, :-1].values  # 11个特征
y_reg = df.iloc[:, -1].values  # 线性回归标签：quality连续值
y_clf = (y_reg > 6).astype(int)  # 逻辑回归标签：>6为1（好酒），否则0

# 数据标准化（梯度下降必须，避免特征尺度差异导致收敛慢）
def standardize(X):
    X = np.array(X, dtype=np.float64)  # 确保是浮点数
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # 防止标准差为0（避免除零错误）
    std[std == 0] = 1e-8
    return (X - mean) / std, mean, std

# 修复变量命名冲突：将标准差变量改为X_std_val
X_std, X_mean, X_std_val = standardize(X)
# 添加偏置项（x0=1）
X_with_bias = np.hstack([np.ones((X_std.shape[0], 1), dtype=np.float64), X_std])

# 划分训练集/测试集（8:2，无sklearn，手动划分）
def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    m = X.shape[0]
    shuffled_idx = np.random.permutation(m)
    test_num = int(m * test_size)
    test_idx = shuffled_idx[:test_num]
    train_idx = shuffled_idx[test_num:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# 划分回归/分类的训练测试集
X_train, X_test, y_reg_train, y_reg_test = train_test_split(X_with_bias, y_reg)
_, _, y_clf_train, y_clf_test = train_test_split(X_with_bias, y_clf)

# ---------------------- 2. 线性回归实现（最小二乘法+梯度下降） ----------------------
# 2.1 最小二乘法（闭式解，直接计算最优theta）
def linear_regression_closed(X, y):
    # theta = (X^T X)^(-1) X^T y
    X_T = X.T
    # 添加正则项避免矩阵奇异（可选，防止不可逆）
    reg = 1e-6 * np.eye(X.shape[1])
    theta = np.linalg.inv(X_T @ X + reg) @ X_T @ y
    return theta

# 2.2 梯度下降法（迭代优化）
def linear_regression_gd(X, y, alpha=0.01, epochs=10000, tol=1e-6):
    m, n = X.shape
    theta = np.zeros(n, dtype=np.float64)
    loss_history = []
    for epoch in range(epochs):
        y_pred = X @ theta
        loss = np.mean((y_pred - y) ** 2) / 2  # MSE/2
        loss_history.append(loss)
        # 计算梯度
        grad = (1/m) * X.T @ (y_pred - y)
        # 更新参数
        theta -= alpha * grad
        # 早停：梯度变化小于阈值则停止
        if np.all(np.abs(alpha * grad) < tol):
            print(f"线性回归梯度下降早停，迭代次数：{epoch+1}")
            break
    return theta, loss_history

# 训练线性回归模型
theta_closed = linear_regression_closed(X_train, y_reg_train)
theta_gd, loss_hist_gd = linear_regression_gd(X_train, y_reg_train, alpha=0.01, epochs=20000)

# 预测
y_reg_pred_closed = X_test @ theta_closed
y_reg_pred_gd = X_test @ theta_gd

# ---------------------- 3. 逻辑回归实现 ----------------------
# sigmoid激活函数
def sigmoid(z):
    # 防止溢出，截断z的范围
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# 逻辑回归梯度下降（降低学习率，避免梯度爆炸）
def logistic_regression_gd(X, y, alpha=0.01, epochs=10000, tol=1e-6):
    m, n = X.shape
    theta = np.zeros(n, dtype=np.float64)
    loss_history = []
    for epoch in range(epochs):
        z = X @ theta
        y_pred = sigmoid(z)
        # 交叉熵损失（添加极小值避免log(0)）
        loss = (-1/m) * np.sum(y * np.log(y_pred + 1e-10) + (1 - y) * np.log(1 - y_pred + 1e-10))
        loss_history.append(loss)
        # 梯度
        grad = (1/m) * X.T @ (y_pred - y)
        # 更新参数
        theta -= alpha * grad
        # 早停
        if np.all(np.abs(alpha * grad) < tol):
            print(f"逻辑回归梯度下降早停，迭代次数：{epoch+1}")
            break
    return theta, loss_history

# 训练逻辑回归模型（调整学习率为0.01，更稳定）
theta_log, loss_hist_log = logistic_regression_gd(X_train, y_clf_train, alpha=0.01, epochs=20000)

# 预测（概率转标签，阈值0.5）
y_clf_pred_prob = sigmoid(X_test @ theta_log)
y_clf_pred = (y_clf_pred_prob >= 0.5).astype(int)

# ---------------------- 4. 模型评估 ----------------------
# 4.1 线性回归评估指标：MSE、RMSE、MAE、R²
def evaluate_regression(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    # R² = 1 - (SS_res / SS_tot)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # 防止分母为0
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2}

# 评估线性回归
eval_closed = evaluate_regression(y_reg_test, y_reg_pred_closed)
eval_gd = evaluate_regression(y_reg_test, y_reg_pred_gd)
print("="*50)
print("线性回归（最小二乘法）评估结果：")
for k, v in eval_closed.items():
    print(f"{k}: {v:.4f}")
print("\n线性回归（梯度下降）评估结果：")
for k, v in eval_gd.items():
    print(f"{k}: {v:.4f}")

# 4.2 逻辑回归评估指标：准确率、精确率、召回率、F1、混淆矩阵
def evaluate_classification(y_true, y_pred):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    # 混淆矩阵
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    # 指标计算（全面防护除零错误）
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Confusion Matrix": np.array([[TN, FP], [FN, TP]])
    }

# 评估逻辑回归
eval_log = evaluate_classification(y_clf_test, y_clf_pred)
print("\n" + "="*50)
print("逻辑回归评估结果：")
for k, v in eval_log.items():
    if k == "Confusion Matrix":
        print(f"{k}:\n{v}")
    else:
        print(f"{k}: {v:.4f}")

# ---------------------- 5. 线性回归&逻辑回归异同点分析 ----------------------
print("\n" + "="*50)
print("线性回归 vs 逻辑回归 异同点分析：")
print("""
【相同点】
1.  本质都是线性模型：模型形式均为 h = θ^T X，核心都是学习参数θ
2.  优化方法一致：都可以用梯度下降法最小化损失函数
3.  都需要特征标准化：避免特征尺度差异影响参数学习
4.  都属于监督学习：需要带标签的训练数据

【不同点】
| 维度                | 线性回归                          | 逻辑回归                          |
|---------------------|-----------------------------------|-----------------------------------|
| 任务类型            | 回归任务（预测连续值）            | 分类任务（预测离散类别/概率）     |
| 输出范围            | (-∞, +∞) 连续实数                 | [0, 1] 概率值                     |
| 激活函数            | 无（直接输出线性组合）             | Sigmoid函数，将线性输出映射到概率 |
| 损失函数            | 均方误差(MSE)                     | 交叉熵损失(Cross-Entropy)         |
| 最优解              | 有闭式解（最小二乘法）            | 无闭式解，必须迭代优化             |
| 假设条件            | 假设y服从正态分布、误差同方差      | 假设标签服从伯努利分布             |
| 评估指标            | MSE、RMSE、R²等回归指标           | 准确率、F1、混淆矩阵等分类指标     |

【核心差异本质】
线性回归直接拟合特征到连续值的映射，逻辑回归通过sigmoid将线性输出转化为概率，解决分类问题，是线性回归的概率化扩展。
""")

# ---------------------- 6. 可视化（损失曲线、预测结果） ----------------------
# 线性回归梯度下降损失曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_hist_gd)
plt.title("线性回归梯度下降损失曲线")
plt.xlabel("迭代次数")
plt.ylabel("MSE/2")

# 逻辑回归梯度下降损失曲线
plt.subplot(1, 2, 2)
plt.plot(loss_hist_log)
plt.title("逻辑回归梯度下降损失曲线")
plt.xlabel("迭代次数")
plt.ylabel("交叉熵损失")
plt.tight_layout()
plt.show()

# 线性回归预测值vs真实值
plt.figure(figsize=(8, 4))
plt.scatter(y_reg_test, y_reg_pred_closed, alpha=0.6, label="闭式解")
plt.scatter(y_reg_test, y_reg_pred_gd, alpha=0.6, label="梯度下降")
plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'r--')
plt.xlabel("真实评分")
plt.ylabel("预测评分")
plt.title("线性回归预测值 vs 真实值")
plt.legend()
plt.show()