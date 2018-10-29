import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split


def read_dataset(fname):
    # 指定第一列为行索引
    data = pd.read_csv(fname, index_col=0)
    # 丢弃无用数据
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
print(train)

y = train['Survived'].values
x = train.drop(['Survived'], axis=1).values  # axis=1表示横轴，方向从左到右；0表示纵轴，方向从上到

print('x:', x)
print('y:', y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print('train dataset:{0}; test dataset:{1}'.format(x_train.shape, x_test.shape))

