import struct

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#   28*28输入矩阵，10个分类，(784,100, 4)
#   激活函数值域大多(0,1), 所以输出结果应以0,1表示，参考数字电路的编码器
class BPNetwork:
    def __init__(self, in_count, hiden_count, out_count, lrate):
        """
        :param in_count:        输入层节点 
        :param hiden_count:     隐藏层节点
        :param out_count:       输出层节点
        :param lrate            学习率
        """
        # 各个层的节点数量
        self.in_count = in_count
        self.hiden_count = hiden_count
        self.out_count = out_count
        self.lrate = lrate
        
        # 十进制与二进制的转化字典
        numberLabel = np.array([[0,0,0,0],[0,0,0,1],
                                [0,0,1,0],[0,0,1,1],
                                [0,1,0,0],[0,1,0,1],
                                [0,1,1,0],[0,1,1,1],
                                [1,0,0,0],[1,0,0,1]])

        # 输入层到隐藏层连线的权重随机初始化
        self.w1 = np.random.randn(self.hiden_count, self.in_count) / \
                np.sqrt(self.in_count) 

        # 隐藏层到输出层连线的权重随机初始化
        self.w2 = np.random.random(self.out_count, self.hiden_count) / \
                np.sqrt(self.hiden_count) 

        # 隐藏层偏置向量
        self.hiden_offset = np.random.randn(self.hiden_count, 1)
        # 输出层偏置向量
        self.out_offset = np.random.randn(self.out_count, 1)


    #   输入与权重的矩阵内积,代入激活函数，计算某层的全部输出
    def feedforward(self, input, weights, biases):
        tmp = np.dot(weights, input) + biases
        activations=[]
        for i in tmp:
            activations.append(sigmoid(i))
        activations = np.array(activations)  
        return tmp, activations

    def NetWork(self, train_data, train_label):
        """
        :param train_data:
        :param train_label:
        :return:
        """
        # 前向传输
        # 储存每层的神经元的值的矩阵，下面循环会 append 每层的神经元的值
        tmp1, activation1 = feedforward(train_data, self.w1, self.hiden_offset)
        tmp2, activation2 = feedforward(activation1, self.w2, self.out_offset)

        #   train_label得转为二进制 
        #   误差
        totalLoss = loss(BinNumber[train_label], activation2)
        #   梯度，结果是含四个元素的数组
        delta = -(BinNumber[train_label]-activation2) * activation2 * (1-activation2)

        #   反向传播,
        backprop(delta, activation2, self.w2)

        backprop( np.dot(self.w2, ),activation1)
        #   梯度下降
    # 4, 4, 4* 30    
    def backprop(self, delta, inputActivations, weights):
        for i in range(0, weights.size):
            weights[:,i] += self.lrate * delta * inputActivations 
        # 更新权值


    #   计算总误差
    def loss(self, train_label, activation):
        loss = (train_label - activation)
        loss = 0.5*loss*loss
        return loss.sum() 


    def train(self, train_img_pattern, train_label):
        if self.in_count != len(train_img_pattern[0]):
            sys.exit("输入层维数与样本维数不等")
        # for num in range(10):
        # for num in range(10):
        for i in range(len(train_img_pattern)):
            # 生成目标向量
            target = train_label
            # for t in range(len(train_img_pattern[num])):
            # 前向传播
            # 隐藏层值等于输入层*w1+隐藏层偏置

            # 反向更新
            error = target - out_value
            # 计算输出层误差
            loss = out_value * (1 - out_value) * error
            # 计算隐藏层误差
            hiden_error = hiden_value * \
                    (1 - hiden_value) * np.dot(self.w2, loss)

            # 更新w2，w2是j行k列的矩阵，存储隐藏层到输出层的权值
            for k in range(self.out_count):
                # 更新w2第k列的值，连接隐藏层所有节点到输出层的第k个节点的边
                # 隐藏层学习率×输入层误差×隐藏层的输出值
                self.w2[:, k] += self.hiden_rate * loss[k] * hiden_value

            # 更新w1
            for j in range(self.hiden_count):
                self.w1[:, j] += self.in_rate * \
                        hiden_error[j] * train_img_pattern[i]

            # 更新偏置向量
            self.out_offset += self.hiden_rate * loss
            self.hiden_offset += self.in_rate * hiden_error


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


sizes = [2, 3, 2]
biases = [np.random.randn(y, 1) for y in sizes[1:]]
print("sizes", sizes[1:])
print(biases)

'''
x = np.arange(0,1, 0.01)
w = 10
b = -5
z = w * x + b
print(z)


# 二维列表与一维列表内积，变成一维列表
w1 = np.random.randn(100, 784) / np.sqrt(784)
w2 = np.random.randn(784)
w3 = np.dot(w1,w2)
print("w3[0]", w3[0])
print("w3[784]", w3[783])
''' 

'''
    元素数量10的数组，训练x，则对应的label[x]=1
    t_label = np.zeros(out_num)
    t_label[label[count]] = 1


    二进制转十进制
    二的4次方为16，所以四位数据可表示16个十进制数
    好处：sigmod的值域（0,1），对应了二进制
'''