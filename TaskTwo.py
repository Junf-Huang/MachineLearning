#   合并文件内容
import os
filedir = os.getcwd()+'/text'
filenames = os.listdir(filedir)
outfile = open('text/result.txt', 'w')
for filename in filenames:
    filepath = filedir+'/' + filename
    for line in open(filepath):
        outfile.writelines(line)
outfile.close()

#   以‘#’为结束符，统计字符个数
str = input('').split("#")[0]
outfile = open('text/result.txt', 'w')
outfile.writelines(str)     # 写入新的一行
outfile.close()
outfile = open('text/result.txt', 'r')
resoult = {}
for line in outfile.readlines():
    print(line)
    for i in line:
        resoult[i] = line.count(i)    
outfile.close()
print(resoult)

#   大小写转换
inputfile = open('text/a.txt', 'r+')
outputfile = open('text/result.txt', 'w+')
for line in inputfile.readlines():
    outputfile.writelines(line.replace(line, line.swapcase()))
inputfile.close()
outputfile.close()

#   整数转换成二进制数,存在二进制文件中
import pickle
dec = int(input('please input a integer:'))
pickle.dump(bin(dec), open('text/bin', 'wb'), pickle.HIGHEST_PROTOCOL)
print(pickle.load(open('text/bin', 'rb')))

#   字典序列化
import pickle
dict = {'Adam': 79, 'Bruno': 90, 'Sam': '64'}
pickle.dump(dict, open('text/score', 'wb'), pickle.HIGHEST_PROTOCOL)
print(pickle.load(open('text/score', 'rb')))

#   异常处理
import sys
b = 0
try:
    a = 2/b
except ZeroDivisionError:
    print("b can not be zero.")
except:
    print("Unexpected error:", sys.exc_info()[0])

#   文件读写异常处理
try:
    file = open('text/ab', 'r')
except FileNotFoundError:
    print("The file doesn't exist.")


#   类
class Person:
    name = ''
    sex = ''
    pid = ''
    email = ''
    phone = ''

    def __init__(self, name, sex, pid):
        self.name = name
        self.sex = sex
        self.pid = pid

    def SetEmail(self, email):
        self.email = email

    def SetPhone(self, phone):
        self.phone = phone
 
    def toString(self):
        print('The name is {}.'.format(self.name))
        print('The sex is {}.'.format(self.sex))
        print('The pid is {}.'.format(self.pid))
        print('The email is {}.'.format(self.email))
        print('The phone is {}.'.format(self.phone))


TestPerson = Person('李一', '男', '223411200010162238')
TestPerson.SetEmail('aaa@mail.com')
TestPerson.toString()


#   学生类
class Student:
    name = ''
    stuSum = ''

    def __init__(self, name, stuSum):
        self.name = name
        self.stuSum = stuSum

    def ShowName(self):
        print('The name is {}.'.format(self.name))

    def ShowSum(self):
        print('The sum is {}.'.format(self.stuSum))


TestStudent = Student('李一', 12)
TestStudent.ShowName()
TestStudent.ShowSum()