# Enumerate (list의 element를 추출할 때 번호를 붙여서 추출)
# no.1
for i, v in enumerate(['tic', 'tac', 'toc']):
    print(i, v)

# no.2
myList = ["a","b","c","d"]
print(list(enumerate(myList))) # [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')]

# no.3 dict로 저장할 수 도 있다.
print({i:j for i,j in enumerate('Mokdong is a city located in South Korea.'.split())})

# Zip (두 개의 list의 값을 병렬적으로 추출함)
# no.1
a_list = ['a1', 'a2', 'a3']
b_list = ['b1', 'b2', 'b3']
for a, b in zip(a_list, b_list):
    print(a,b)

# no.2
a,b,c = zip((1,2,3),(10,20,30), (100,200,300))
print(a,b,c)

print([sum(x) for x in zip((1,2,3), (10,20,30), (100,200,300))])

# no.3
a_list = ['a1', 'a2', 'a3']
b_list = ['b1', 'b2', 'b3']

for i, (a,b) in enumerate(zip(a_list, b_list)):
    print(i, a,b)

