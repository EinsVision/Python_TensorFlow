# Asterisk *
# 흔히 알고 있는 * 를 의미함
# 단순 곱셈, 제곱연산, 가변 인자 활용 등 다양하게 사용됨
# * Asterisk
def asterisk_test(a, *args): # *args 한번에 받으라는 의미이다.
    print(a, args)
    print(type(args))

asterisk_test(1,2,3,4,5,6) # 여러개의 값을 한번에 받으려고 사용한다.
# 1 (2, 3, 4, 5, 6)
# <class 'tuple'>

# keyword 인자는 ** 를 사용한다. keyword를 넘겨울 때 **를 사용하자.
def asterist_test1(a, **kargs):
    print(a, kargs)
    print(type(kargs))

asterist_test1(1,b=2 ,c=3,d=4,e=5,f=6)
# 1 {'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}
# <class 'dict'>

# unpacking 
def asterisk_test2(a, args):
    print(a, *args)
    print(type(args))

asterisk_test2(1, (2,3,4,5,6)) # tuple의 형태로 하나의 자료형이 넘겨졌지만,
                               # unpacking해서 출력하고 있다.
# 1 2 3 4 5 6
# <class 'tuple'>

a, b, c = ([1,2], [3,4], [5,6])
print(a,b,c)
print(type(a))

data = ([1,2], [3,4], [5,6])
print(*data) # 이렇게 * 붙이면 한번에 출력이 된다.
print(data)

# [1, 2] [3, 4] [5, 6]         : print(*data)
# ([1, 2], [3, 4], [5, 6])     : print(data)

for data in zip(*([1,2], [3,4], [5,6] )):
    print(sum(data))

