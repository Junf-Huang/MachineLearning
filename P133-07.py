import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus


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


train = read_dataset('datasets/titanic/train.csv')
print("dataHead->", train.head())

y = train['Survived'].values
X = train.drop(['Survived'], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print('train dataset: {0}; test dataset: {1}'.format(X_train.shape,
                                                     X_test.shape))

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print('Without parameter:\ntrain score: {0}; test score: {1}'.format(
    train_score, test_score))

dot_data = export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("titani0.png")

# 1. 在电脑上安装 graphviz
# 3. 在当前目录查看生成的决策树 titanic.png


# 参数选择 max_depth
def cv_score_depth(d):
    clf = DecisionTreeClassifier(max_depth=d)
    clf.fit(X_train, y_train)
    tr_score = clf.score(X_train, y_train)
    cv_score = clf.score(X_test, y_test)
    return (tr_score, cv_score)


depths = range(2, 15)
scores = [cv_score_depth(d) for d in depths]
tr_scores = [s[0] for s in scores]
cv_scores = [s[1] for s in scores]  # 测试集分数

best_score_index = np.argmax(cv_scores)
best_score = cv_scores[best_score_index]
best_param = depths[best_score_index]  # 最优参数的索引与最高分数的索引相同
print('best max_depth param: {0}; best score: {1}'.format(
    best_param, best_score))

plt.figure(num='max_depth', figsize=(10, 6), dpi=120)
plt.grid()
plt.xlabel('max depth of decision tree')
plt.ylabel('score')
plt.plot(depths, cv_scores, '.g-', label='cross-validation score')
plt.plot(depths, tr_scores, '.r--', label='training score')
plt.legend()
plt.show()


# 训练模型，并计算评分
def cv_score_purity(val):
    clf = DecisionTreeClassifier(criterion='gini', min_impurity_decrease=val)
    clf.fit(X_train, y_train)
    tr_score = clf.score(X_train, y_train)
    cv_score = clf.score(X_test, y_test)
    return (tr_score, cv_score)


# 指定参数范围，分别训练模型，并计算评分
values = np.linspace(0, 0.1, 30)
scores = [cv_score_purity(v) for v in values]
tr_scores = [s[0] for s in scores]
cv_scores = [s[1] for s in scores]

# 找出评分最高的模型参数
best_score_index = np.argmax(cv_scores)
best_score = cv_scores[best_score_index]
best_param = values[best_score_index]
print('best min_impurity param: {0}; best score: {1}'.format(
    best_param, best_score))

# 画出模型参数与模型评分的关系
plt.figure(num="min_impurity_decrease", figsize=(10, 6), dpi=144)
plt.grid()
plt.xlabel('threshold of entropy')
plt.ylabel('score')
plt.plot(values, cv_scores, '.g-', label='cross-validation score')
plt.plot(values, tr_scores, '.r--', label='training score')
plt.legend()
plt.show()


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


thresholds = np.linspace(0, 0.1, 30)
# Set the parameters by cross-validation
param_grid = {'min_impurity_decrease': thresholds}

clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)  # cv数据划分份数
clf.fit(X, y)  # X,y为原始数据集
print("best min_impurity param: {0}\nbest score: {1}".format(
    clf.best_params_, clf.best_score_))

plot_curve(thresholds, clf.cv_results_, xlabel='gini thresholds')

entropy_thresholds = np.linspace(0, 1, 50)
gini_thresholds = np.linspace(0, 0.5, 50)

# Set the parameters by cross-validation
param_grid = [{
    'criterion': ['entropy'],
    'min_impurity_decrease': entropy_thresholds
}, {
    'criterion': ['gini'],
    'min_impurity_decrease': gini_thresholds
}, {
    'max_depth': range(2, 10)
}, {
    'min_samples_split': range(2, 30, 2)
}]

clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
clf.fit(X, y)
print("best multiple param: {0}\nbest score: {1}".format(
    clf.best_params_, clf.best_score_))

# entropy, 生成dot文件
clf = DecisionTreeClassifier(
    criterion='entropy', min_impurity_decrease=0.53061224489795911)
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print('entropy\ntrain score: {0}; test score: {1}'.format(
    train_score, test_score))

# 导出 titanic.dot 文件
with open("titanic.dot", 'w') as f:
    f = export_graphviz(clf, out_file=f)

# 1. 在电脑上安装 graphviz
# 2. 运行 `dot -Tpng titanic.dot -o titanic.png`
# 3. 在当前目录查看生成的决策树 titanic.png
