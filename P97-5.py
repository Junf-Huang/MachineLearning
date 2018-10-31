import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from matplotlib.figure import SubplotParams

# 初始化 X,Y 的值域 
n_dots = 200
X = np.linspace(-20, 20, n_dots)
Y = np.sin(X) + 0.2 * np.random.rand(n_dots) - 0.1
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)


# 模型的初始化
def polynomial_model(degree=1):
    # degree：多项式的阶数, bias：偏差
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression(normalize=True)
    # pipeline类似于机器学习当中的管道，数据经过这个管道的处理之后，就会返回归整化的数据
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    return pipeline


# 模型的训练与评分
degrees = [2, 3, 5, 10]
results = []
for d in degrees:
    model = polynomial_model(degree=d)
    # fit() 训练模型的接口，有监督学习
    model.fit(X, Y)
    # score 得分越高越好
    train_score = model.score(X, Y)
    # predict() 聚类分析，把新数据归入某个聚类里
    mse = mean_squared_error(Y, model.predict(X))
    results.append({
        "model": model,
        "degree": d,
        "score": train_score,
        "mse": mse
    })
for r in results:
    print("degree: {}; train score: {}; mean squared error: {}".format(
        r["degree"], r["score"], r["mse"]))


# dpi 图布大小，笔记本约100
plt.figure(
    num='firstFigure',
    figsize=(12, 6),
    dpi=100,
    subplotpars=SubplotParams(hspace=0.3))
for i, r in enumerate(results):
    # 切换子图
    fig = plt.subplot(2, 2, i + 1)
    # 限制x轴长度
    plt.xlim(-20, 20)
    plt.title("LinearRegression degree={}".format(r["degree"]))
    # 散点图
    plt.scatter(X, Y, s=5, c='b', alpha=0.5)
    plt.plot(X, r["model"].predict(X), 'r-')
    plt.plot(X, np.sin(X), 'g--')
plt.show()
