# Ch02. Python data 처리 ( 1. 제어문 )
# 1. 제어문
# 2. 반복문
# 3. 함수와 클래스
# 4. 데이터형
# 5. Numpy
# 6. Pandas

# 제어문 
myscore = input('성적은 ') # 어떤 값을 받을 때 쓰는 input box
myscore = int(myscore) # input box는 text 형식이라서 int()가 필요하다.

if(myscore >= 80 and myscore < 90):
    print("잘했어요")
elif(myscore >=90):
    print("놀랐어요")
else:
    print("못했어요.")