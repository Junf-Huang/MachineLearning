import struct

import numpy as np
import matplotlib.pyplot as plt

def loadLabelSet(fname):
    with open(fname, 'rb') as f1:
        buf = f1.read()
        head = struct.unpack_from('>II', buf, 0)  # 取前2*4个字节，返回一个元组
        labelNum = head[1]
        #   图片的数量，宽度，高度
        print("labelNum", labelNum)
        offset = struct.calcsize('>II')  # 定位到data开始的位置

        #   每1B为一标签
        bits = labelNum
        bitsString = '>' + str(bits) + 'B'  # fmt格式：'>4B'
        lables = struct.unpack_from(bitsString, buf, offset)
    return lables


def loadImageSet(fname):
    with open(fname, 'rb') as f1:
        buf = f1.read()
        head = struct.unpack_from('>IIII', buf, 0)  # 取前4×4个字节，返回一个元组,一个I对应四位字节
        imgNum = head[1]
        width = head[2]
        height = head[3]
        #   图片的数量，宽度，高度
        print(imgNum, " ", width, " ", height)
        offset = struct.calcsize('>IIII')  # 定位到data开始的位置

        #   每一个784B为一张图片
        bits = imgNum * width * height  # data一共有60000*28*28个像素值
        bitsString = '>' + str(bits) + 'B'  # fmt格式：'>784B'
        imgs = struct.unpack_from(bitsString, buf, offset)
        #   显示图像，二维数组(imgNum, with, height)
        #   方便模型训练，一维数组(imgNum, with*height)
        im = np.reshape(imgs, (imgNum, width * height))
    return im


class BPNetwork:
    def __init__(self, in_count, hiden_count, out_count, lrate):
        """
        :param in_count:        输入层节点 
        :param hiden_count:     隐藏层节点
        :param out_count:       输出层节点
        :param lrate            学习率
        """

        # 字典
        self.numberLabel = np.zeros(out_count)

        # 各个层的节点数量
        self.in_count = in_count
        self.hiden_count = hiden_count
        self.out_count = out_count
        self.lrate = lrate

        # 输入层到隐藏层连线的权重随机初始化
        self.w1 = np.random.randn(self.hiden_count, self.in_count) / np.sqrt(
            self.in_count)
    
        # 隐藏层到输出层连线的权重随机初始化
        self.w2 = np.random.randn(self.out_count, self.hiden_count) / np.sqrt(
            self.hiden_count)

        # 隐藏层偏置向量,因为点乘得出的矩阵为纵列，所以这里设置为纵列列表
        self.hiden_offset = np.random.randn(self.hiden_count)
        # 输出层偏置向量
        self.out_offset = np.random.randn(self.out_count)

    #   输入与权重的矩阵内积,代入激活函数，计算某层的全部输出
    def feedforward(self, input, weights, biases):
        return sigmoid(np.dot(weights, input) + biases)

    def train(self, train_data, train_label):
        """
        :param train_data:
        :param train_label:
        :return:
        """
        self.numberLabel[train_label] = 1
        # 特征向量归一化
        train_data = train_data / 256 
        # 前向传输
        # 储存每层的神经元的值的矩阵，下面循环会 append 每层的神经元的值
        activation1 = self.feedforward(train_data, self.w1, self.hiden_offset)
        activation2 = self.feedforward(activation1, self.w2, self.out_offset)
        #   均方差
        totalLoss = self.loss(self.numberLabel, activation2)
        #   对应元素相乘,np.multiply()或 *
        loss = self.numberLabel - activation2
        print("out:", np.argmax(activation2), " num:", train_label)
        delta1 = loss * activation2 * (1 - activation2)
        delta2 = np.dot(delta1, self.w2) * activation1 * (1 - activation1)
        #   反向传播,
        for i in range(0, delta1.size):
            #   梯度下降公式
            self.w2[i] += self.lrate * delta1[i] * activation1

        for i in range(0, delta2.size):
            self.w1[i] += self.lrate * delta2[i] * train_data

        self.out_offset += self.lrate * delta1
        self.hiden_offset += self.lrate * delta2

        self.numberLabel[train_label] = 0

    def backprop(self, delta, inputActivations, weights):
        for i in range(0, delta.size):
            #   梯度下降公式
            weights[i] -= self.lrate * delta[i] * inputActivations
        # 更新权值

    #   成本函数，计算总误差
    def loss(self, train_label, activation):
        loss = (train_label - activation)
        loss = 0.5 * loss * loss
        return loss.sum()

    def run(self, imgs, labels, size):
        count = 0
        for i, j in zip(imgs, labels):
            i = i / 256
            activation1 = self.feedforward(i, self.w1, self.hiden_offset)
            activation2 = self.feedforward(activation1, self.w2,
                                           self.out_offset)
            if (np.argmax(activation2) == j):
                count += 1
        return count / size


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


labels = loadLabelSet('datasets/mnist/train-labels-idx1-ubyte')
# print("labels", labels[:300])

imgs = loadImageSet('datasets/mnist/train-images-idx3-ubyte')
# print("imgs[0]", imgs[:300])


TestNetwork = BPNetwork(784, 30, 10, 0.2)
count = 10000
for i, j in zip(imgs[:count], labels[:count]):
    TestNetwork.train(i, j)

offset = 100
accuracy = TestNetwork.run(imgs[count:count + offset],
                           labels[count:count + offset], offset)
print("accuracy", accuracy)

np.savetxt("w1.txt", TestNetwork.w1)
np.savetxt("w2.txt", TestNetwork.w2)
'''
图片显示，得先修改loadImageSet()的im矩阵
plt.title("label: "+ str(labels[1]))
plt.imshow(imgs[1], cmap='gray')
plt.show()
'''
