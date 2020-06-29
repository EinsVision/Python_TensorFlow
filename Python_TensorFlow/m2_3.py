
# Ch02. Python data 처리 ( 4. 데이터형, 5. Numpy )
# 1. 제어문
# 2. 반복문
# 3. 함수와 클래스
# 4. 데이터형
# 5. Numpy
# 6. Pandas

# 데이터형
# python 데이터형
# 리스트 (list) num = [1,2,3] 데이터 수정 삭제 ㅇ, index ㅇ (하나의 열이라고 볼 수 있다.)
# 튜플 (tuple)  num = (1,2,3) 데이터 수정 삭제 X, index ㅇ
# 세트 (set)    num = {1,2,3} 데이터 수정 삭제 ㅇ, index X (값이 순서대로 저장되어 있다.)
# 딕셔너리 (dictionary)  dic = {'name':'pey', 'phone':'0119993323', 'birth': '1118'}
import numpy as np

# import numpy as np
num1 = [1,2,3,4]
num2 = [5,6,7,8]
num = [num1, num2]
print(num) # [[1, 2, 3, 4], [5, 6, 7, 8]]

list1 = [1,2,3,4] # list ( 4x1 ) 열 vector이다.
a = np.array(list1) # list를 배열로 바꿔준 형태이다.
print(a.shape)
print(a.sum())
print(a.std())

b = np.array([[1,2,3],[4,5,6]]) # ( 2x3 ) vector이다.
# array([[1,2,3],
#        [4,5,6]])     ( 2x3 ) 

print(b.shape)
print(b[0,0])
