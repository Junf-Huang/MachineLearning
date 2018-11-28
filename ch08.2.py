import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.datasets import make_blobs

#  画出样本点
def plot_hyperplane(clf, X, y, h=0.02, draw_sv=True, title='hyperplan'):
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    plt.title(title)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='hot', alpha=0.5)

    markers = ['o', 's', '^']
    colors = ['b', 'r', 'c']
    labels = np.unique(y)
    for label in labels:
        plt.scatter(
            X[y == label][:, 0],
            X[y == label][:, 1],
            c=colors[label],
            marker=markers[label])
    if draw_sv:
        sv = clf.support_vectors_
        plt.scatter(sv[:, 0], sv[:, 1], c='y', marker='x')


X, y = make_blobs(n_samples=100, centers=3, random_state=0, cluster_std=0.8)
'''
clf_linear1 = svm.SVC(C=0.01, kernel='linear')
clf_linear2 = svm.SVC(C=0.1, kernel='linear')
clf_linear3 = svm.SVC(C=1.0, kernel='linear')
clf_linear4 = svm.SVC(C=10.0, kernel='linear')
clf_poly1 = svm.SVC(C=1.0, kernel='poly', degree=1)
clf_poly2 = svm.SVC(C=1.0, kernel='poly', degree=2)
clf_poly3 = svm.SVC(C=1.0, kernel='poly', degree=3)
clf_poly4 = svm.SVC(C=1.0, kernel='poly', degree=4)
'''

clf_rbf0 = svm.SVC(C=1.0, kernel='rbf', gamma=0.1)
clf_rbf1 = svm.SVC(C=1.0, kernel='rbf', gamma=0.5)
clf_rbf2 = svm.SVC(C=1.0, kernel='rbf', gamma=1.0)
clf_rbf3 = svm.SVC(C=1.0, kernel='rbf', gamma=5.0)

plt.figure(figsize=(10, 10), dpi=120)

clfs = [clf_rbf0, clf_rbf1, clf_rbf2, clf_rbf3]
titles = [
    'Gaussian Kernel-1', 'Gaussian Kernel-2',
    'Gaussian Kernel-3', 'Gaussian Kernel-4'
]
for clf, i in zip(clfs, range(len(clfs))):
    clf.fit(X, y)
    plt.subplot(2, 2, i + 1)
    plot_hyperplane(clf, X, y, title=titles[i])
plt.show()


'''
plt.figure(figsize=(10, 10), dpi=120)

clfs = [clf_linear1, clf_linear2, clf_linear3, clf_linear4]
titles = [
    'Linear Kernel-1', 'Linear Kernel-2',
    'Linear Kernel-3', 'Linear Kernel-4'
]
for clf, i in zip(clfs, range(len(clfs))):
    clf.fit(X, y)
    plt.subplot(2, 2, i + 1)
    plot_hyperplane(clf, X, y, title=titles[i])
plt.show()
'''