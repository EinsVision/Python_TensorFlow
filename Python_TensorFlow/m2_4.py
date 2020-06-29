# Ch02. Python data 처리 ( 6. Pandas )
# 1. 제어문
# 2. 반복문
# 3. 함수와 클래스
# 4. 데이터형
# 5. Numpy
# 6. Pandas

# Pandas
import pandas as pd

# 데이터 가져오기 (data frame)
df = pd.read_csv('grade.csv')

df = pd.DataFrame(df)
print(df.head()) # 상위 5개만을 가져온다.

print(df.shape) # (50, 4) 
print(df.index) # RangeIndex(start=0, stop=50, step=1)
print(df.columns) # Index(['id', 'msex', 'csex', 'grade'], dtype='object')
print(df.values) # 행 렬 로 이루어져 있음

# dataframe에서 행 선택하기
print(df.iloc[5])   

# dataframe에서 열 선택하기
print(df.iloc[:,1])

df = df.drop('id', axis=1)
print(df.head())