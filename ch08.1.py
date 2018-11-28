import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (GridSearchCV, ShuffleSplit,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.svm import SVC, LinearSVC

from common.utils import plot_learning_curve, plot_param_curve

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
print('data shape: {0}; no. positive: {1}; no. negative: {2}'.format(
    X.shape, y[y == 1].shape[0], y[y == 0].shape[0]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = SVC(C=1.0, kernel='rbf', gamma=0.1)
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print('train score: {0}; test score: {1}'.format(train_score, test_score))

gammas = np.linspace(0, 0.0003, 30)
param_grid = {'gamma': gammas}
clf = GridSearchCV(SVC(), param_grid, cv=5)
clf.fit(X, y)
print("best param: {0}\nbest score: {1}".format(clf.best_params_,
                                                clf.best_score_))
plt.figure(figsize=(10, 4), dpi=144)
plot_param_curve(
    plt, gammas, clf.cv_results_, xlabel='gamma')
plt.show()


cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
title = 'Learning Curves for Gaussian Kernel'

plt.figure(figsize=(10, 4), dpi=144)
plot_learning_curve(
    plt,
    SVC(C=1.0, kernel='rbf', gamma=0.01),
    title,
    X,
    y,
    ylim=(0.5, 1.01),
    cv=cv)
plt.show()

clf = SVC(C=1.0, kernel='poly', degree=2)
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print('train score: {0}; test score: {1}'.format(train_score, test_score))


def create_model(degree=2, **kwarg):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    scaler = MinMaxScaler()
    linear_svc = LinearSVC(**kwarg)
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("scaler", scaler), ("linear_svc", linear_svc)])
    return pipeline


clf = create_model(penalty='l1', dual=False)
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print('train score: {0}; test score: {1}'.format(train_score, test_score))


cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
title = 'Learning Curves for LinearSVC with Degree={0}'
degrees = [1, 2]

plt.figure(figsize=(12, 4), dpi=144)
for i in range(len(degrees)):
    plt.subplot(1, len(degrees), i + 1)
    plot_learning_curve(
        plt,
        create_model(penalty='l1', dual=False, degree=degrees[i]),
        title.format(degrees[i]),
        X,
        y,
        ylim=(0.8, 1.01),
        cv=cv)
plt.show()
