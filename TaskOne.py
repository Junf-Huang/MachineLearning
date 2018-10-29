import random as rd
import support

list1 = [1, 2, 3, 4, 5, 6, 7]
print("list: ", list1)

L1 = set('1,2,3 4')
L2 = set('5 6 7 8')
print("L1: ",  L1)
print("L2: ", L2)
print("L1 & L2: ", L1 & L2)
print("L1 - L2: ", L1 - L2)
print("L2 - L1: ", L2 - L1)

count = 0
classmate = {}
while count < 5:
    stuName = input("Name：")
    stuGrade = input("Grade: ")
    classmate[stuName] = stuGrade
    count = count + 1
print("classmante:", classmate)
nameList = list(classmate.keys())
gradeList = list(classmate.values())
print("nameList:", nameList)
print("gradeList:", gradeList)

tel = {'jack': 4098, 'sape': 4139, 'guido': 4127}
key = input("key：")
if key in tel:
    print("value:", tel[key])
else:
    print("The key you input doesn't exist!")

count = 0
countDict = {}
while count < 1000:
    ranNum = rd.randint(1, 50)
    if ranNum in countDict:
        countDict[ranNum] += 1
    else:
        countDict[ranNum] = 1
    count += 1
print("countDict:", countDict)

#温度
def convert_celsius(fahrenheit):
    return (5*(fahrenheit-32)/9)


fahrenheit = float(input('输入华氏温度: '))
print("摄氏温度:", convert_celsius(fahrenheit))

#质数
def judge_prime(num):
    if num == 1:
        print(num, "is not a prime.")
        return 0 
    if num == 2:
        print(num, "is a prime.")
        return 0
    if num > 2:
        for i in range(2, num):
            if (num % i) == 0:
                print(num, "is not a prime.")
                break
        print(num, "is a prime.")
    else:
        print(num, "is not a prime.")


num = int(input("input a number: "))
judge_prime(num)


def Max_Min(x, y):
    if x < y:
        x, y = y, x
    max = 1
    tiple = x*y
    for i in range(2, y+1):
        if(y % i == 0 and x % i == 0):
            max = i
    print("The max common divisor is", max)
    print("The min common multiple is", (tiple/max))            


x = int(input('please input a integer:'))
y = int(input('please input a integer:'))
Max_Min(x, y)


def is_palindrome(x):
    x = str(x)
    if x == x[::-1]:
        print('%s is a palindrome.' % x)
    else:
        print('%sis not a palindrom.' % x)


x = input('please input a integer:')
is_palindrome(x)


def recursive_add(n):
    if(n == 1):
        return 1
    else:
        return recursive_add(n-1) + n


n = int(input('please input a integer:'))
print("sum is %s." % recursive_add(n))

support.convert_celsius(100.4)
support.convert_fahrenheit(38)

#   x的y次幂
def getPower(x, y):
    a = x
    while y > 1:
        x = x * a
        y = y - 1
    return x


x = int(input("input a number: "))
y = int(input("input a number: "))
print(getPower(x, y))


#   fiber
def fib_re(n):
    if(n == 0 or n == 1):
        return 1
    else:
        return fib_re(n-1) + fib_re(n-2)
 
 
n = int(input("input a number: "))
for i in range(n):
    print(fib_re(i))