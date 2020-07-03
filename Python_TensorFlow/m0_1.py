colors = ['red', 'blue', 'green', 'yellow' ]
# no.1 일반코드( list 합치기 )
result = ''
for i in colors :
    result += i

print(result)

# no.2 pythonic code ( list 합치기 )
# join 함수
# string list를 합쳐 하나의 string으로 반환할 때 사용

result = ''
result = ''.join(colors)
print(result)

result = ''
result = ','.join(colors)
print(result)

result = ''
result = '-'.join(colors)
print(result)

# split 함수 
# string type의 값을 나눠서 List 형태로 반환
items = 'zero one two three'.split() # 빈칸을 기준으로 문자열 나누기
print(items) # ['zero', 'one', 'two', 'three']

example = 'python, jquuery, javascript'
print(example.split(","))

# List에 있는 각 값을 a,b,c 변수로 Unpacking
a,b,c = example.split(",")
print(a)
print(b)
print(c)

# List comprehensions
# no.1 
result = []
for i in range(10):
    result.append(i)

print(result)

# no.2
result = [i for i in range(10)]
print(result)

# no.3 (조건을 넣을 수도 있다.)
result = [i for i in range(10) if i % 2  == 0]
print(result)

# no.4 for loop을 2개 사용하는 경우
word_1 = "Hello"
word_2 = "World"
result = [i+j for i in word_1 for j in word_2]

print(result)

# no.5 sort 함수
case_1 = ["A", "B", "C"]
case_2 = ["D", "E", "A"]
result = [i+j for i in case_1 for j in case_2 if not(i==j)]
print(result)

result.sort()
print(result)

# no. 6 2차원 list
words = 'The quick brown fox jumps over the lazy dog'.split()
print(words)

stuff = [[w.upper(), w.lower(), len(w)] for w in words]

for i in stuff:
    print(i)