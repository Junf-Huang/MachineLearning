import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from common.utils import plot_learning_curve


def polynomial_model(degree=1):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression(normalize=True)
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    return pipeline

boston = load_boston()
X = boston.data
y = boston.target
print("数组信息：", X.shape)
print("索引一：", X[0])
print("波士顿特征：", boston.feature_names)

# 普通模型训练
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3)
model = LinearRegression()
start = time.perf_counter()
model.fit(X_train, y_train)
# 模型针对训练集的得分
train_score = model.score(X_train, y_train)
# 模型针对测试集的得分
cv_score = model.score(X_test, y_test)
print('elaspe1: {0:.6f}; train_score: {1:0.6f}; cv_score: {2:.6f}'.format(
    time.perf_counter() - start, train_score, cv_score))

# 具备线性特征的模型训练
model = polynomial_model(degree=5)
start = time.perf_counter()
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
cv_score = model.score(X_test, y_test)
print('elaspe2: {0:.6f}; train_score: {1:0.6f}; cv_score: {2:.6f}'.format(
    time.perf_counter() - start, train_score, cv_score))


# 含有交叉验证的模型训练
# 交叉验证(test_size代表test集所占比例)
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
degrees = [5]

start = time.perf_counter()
plt.figure(figsize=(12, 6), dpi=100)
title = 'Learning Curves (degree={0})'
for i in range(len(degrees)):
    plt.subplot(1, 3, i + 1)
    plot_learning_curve(
        plt,
        polynomial_model(degrees[i]),
        title.format(degrees[i]),
        X,
        y,
        ylim=(0.01, 1.01),
        cv=cv)

print('elaspe3: {0:.6f}'.format(time.perf_counter() - start))
plt.show()
