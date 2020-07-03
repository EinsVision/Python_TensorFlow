# Lambda (함수 이름 없이, 함수처럼 쓸 수 있는 익명함수)
### 일반 함수
# def f(x,y):
#    return x + y
# print(f(3,4))

### Lambda 함수
# f = lambda x,y : x + y
# print(f(1,4))

f = lambda x,y : x + y
print(f(1,4)) 

# map function
# sequence 자료형 각 element에 동일한 function을 적용함
# map(function_name, list_data)
ex = [1, 2, 3, 4, 5]
f = lambda x: x ** 2
print(list(map(f, ex))) # [1, 4, 9, 16, 25]

ex = [1,2,3,4,5]
f = lambda x,y : x+y
print(list(map(f,ex,ex))) # [2, 4, 6, 8, 10]

print(list(map(lambda x: x ** 2 if x % 2 == 0 else x,ex)))
# [1, 4, 3, 16, 5]

# reduce function
# map function과 달리 list에 똑같은 함수를 적용해서 통합

from functools import reduce
print(reduce(lambda x, y : x + y, [1, 2, 3, 4, 5]))
# 1 + 2 / 3 + 3 / 6 + 4 / 10 + 5 ( 이렇게 계산해 나가는 것이 reduce 함수)

def factorial(n):
    return reduce(lambda x, y : x*y, range(1, n+1))

print(factorial(5))