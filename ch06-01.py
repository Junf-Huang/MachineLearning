import matplotlib.pyplot as plt
import numpy as np


def f_1(x):
    return -np.log(x)


def f_0(x):
    return -np.log(1 - x)


X = np.linspace(0.01, 0.99, 100)
f = [f_1, f_0]
titles = ["y=1: $-log(h_\\theta(x))$", "y=0: $-log(1 - h_\\theta(x))$"]
plt.figure(figsize=(12, 4), dpi=100)
for i in range(len(f)):
    plt.subplot(1, 2, i + 1)
    plt.title(titles[i])
    plt.xlabel("$h_\\theta(x)$")
    plt.ylabel("$Cost(h_\\theta(x), y)$")
    plt.plot(X, f[i](X), 'r-')


def L1(x):
    return 1 - np.abs(x)


def L2(x):
    return np.sqrt(1 - np.power(x, 2))


def contour(v, x):
    return 5 - np.sqrt(v - np.power(x + 2, 2))  # 4x1^2 + 9x2^2 = v

# 格式化坐标轴
def format_spines(title):
    ax = plt.gca()  # gca 代表当前坐标轴，即 'get current axis'
    ax.spines['right'].set_color('none')  # 隐藏坐标轴
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')  # 设置刻度显示位置
    ax.spines['bottom'].set_position(('data', 0))  # 设置下方坐标轴位置
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))  # 设置左侧坐标轴位置

    plt.title(title)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)


plt.figure(figsize=(8.4, 4), dpi=100)

x = np.linspace(-1, 1, 100)
cx = np.linspace(-3, 1, 100)

plt.subplot(1, 2, 1)
format_spines('L1 norm') # 图像名
plt.plot(x, L1(x), 'r-', x, -L1(x), 'r-')
plt.plot(cx, contour(20, cx), 'r--', cx, contour(15, cx), 'r--', cx,
         contour(10, cx), 'r--')

plt.subplot(1, 2, 2)
format_spines('L2 norm')
plt.plot(x, L2(x), 'b-', x, -L2(x), 'b-')
plt.plot(cx, contour(19, cx), 'b--', cx, contour(15, cx), 'b--', cx,
         contour(10, cx), 'b--')
plt.show()
