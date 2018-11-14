import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


def read_dataset(fname):
    # 指定第一列作为行索引
    data = pd.read_csv(fname, index_col=0)
    # 丢弃无用的数据
    data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    # 处理性别数据
    data['Sex'] = (data['Sex'] == 'male').astype('int')
    # 处理登船港口数据
    labels = data['Embarked'].unique().tolist()
    data['Embarked'] = data['Embarked'].apply(lambda n: labels.index(n))
    # 处理缺失数据
    data = data.fillna(0)
    return data


def plot_curve(train_sizes, cv_results, xlabel):
    train_scores_mean = cv_results['mean_train_score']
    train_scores_std = cv_results['std_train_score']
    test_scores_mean = cv_results['mean_test_score']
    test_scores_std = cv_results['std_test_score']
    plt.figure(figsize=(10, 6), dpi=144)
    plt.title('parameters turning')
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('score')
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r")
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g")
    plt.plot(
        train_sizes,
        train_scores_mean,
        '.--',
        color="r",
        label="Training score")
    plt.plot(
        train_sizes,
        test_scores_mean,
        '.-',
        color="g",
        label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()


train = read_dataset('datasets/titanic/train.csv')
print("dataHead->", train.head())

y = train['Survived'].values
X = train.drop(['Survived'], axis=1).values

thresholds = np.arange(2, 50, 1)
# Set the parameters by cross-validation
param_grid = {'min_samples_split': thresholds}

clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)  # cv数据划分份数
clf.fit(X, y)  # X,y为原始数据集
print("best min_samples param: {0}\nbest score: {1}".format(
    clf.best_params_, clf.best_score_))

plot_curve(thresholds, clf.cv_results_, xlabel='samples thresholds')
