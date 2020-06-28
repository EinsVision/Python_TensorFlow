
# Ch02. Python data 처리 ( 2. 반복문, 3. 함수와 클래스)
# 1. 제어문
# 2. 반복문
# 3. 함수와 클래스
# 4. 데이터형
# 5. Numpy
# 6. Pandas

# 반복문
cal = 0 
for i in range(101): # 0부터 시작한다. 그리고 101 앞번호까지 반복한다.
    cal += i
print('1부터 100까지 더한 값은 ',cal)

# 리스트 이용
test = [1,2,3,4]
for i in test: # 리스트 개수만큼 반복시킬 수 있다.
    print('리스트 이용: ',i)

# 인덱스 확인이 필요할 경우
test2 = [5,6,7,8]
for i in enumerate(test2):
    print('인덱스 확인: ',i)

# 함수를 정의하는 방법 (반환값이 없는 함수)
def avg(x,y):
    print((x+y)/2)

# 함수를 정의하는 방법 (반환값이 있는 함수)
def avg1(x,y):
    return((x+y)/2)

avg(4,5)

a = avg1(7,5)
print(a)

# class 정의하는 방법
class Cal():
    def __init__(self, x1, x2): # init: 초기화 세팅 (class를 여기서 부터 시작해라.)
        self.x1 = x1
        self.x2 = x2

    def add(self):
        result = self.x1 + self.x2
        return result

    def sub(self):
        result = self.x1 - self.x2
        return result

cal1 = Cal(4,3)

print(cal1.add())
print(cal1.sub())
    