import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from common.utils import plot_learning_curve

# 载入数据
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
print('data shape: {0}; no. positive: {1}; no. negative: {2}'.format(
    X.shape, y[y == 1].shape[0], y[y == 0].shape[0]))
print(cancer.data[0])
print(cancer.feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 增加多项式预处理
def polynomial_model(**kwarg):
    polynomial_features = PolynomialFeatures()
    logistic_regression = LogisticRegression(**kwarg)
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("logistic_regression", logistic_regression)])
    return pipeline


model1 = polynomial_model(penalty='l1')
model1.fit(X_train, y_train)

train_score = model1.score(X_train, y_train)
cv_score = model1.score(X_test, y_test)

logistic_regression = model1.named_steps['logistic_regression']
print('model-1 parameters shape: {0}; count of non-zero element: {1}'.format(
    logistic_regression.coef_.shape,
    np.count_nonzero(logistic_regression.coef_)))

model2 = polynomial_model(penalty='l2')
model2.fit(X_train, y_train)

train_score = model2.score(X_train, y_train)
cv_score = model2.score(X_test, y_test)

logistic_regression = model2.named_steps['logistic_regression']
print('model-2 parameters shape: {0}; count of non-zero element: {1}'.format(
    logistic_regression.coef_.shape,
    np.count_nonzero(logistic_regression.coef_)))