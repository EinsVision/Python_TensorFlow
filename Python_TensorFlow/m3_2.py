# Ch03. 데이터 가져오기/전처리/테스트용 데이터 분리

# 1. 기본 package
import numpy as np # numpy 패키지 가져오기
import matplotlib.pyplot as plt # 시각화 패키지 가져오기

# 2. 데이터 가져오기
import pandas as pd # csv->dataframe으로 전환
from sklearn import datasets # python 저장 데이터 가져오기

# 3. 데이터 전처리 (표준화 시킨다.)
from sklearn.preprocessing import StandardScaler # 연속변수의 표준화
from sklearn.preprocessing import LabelEncoder # 번주형 변수 수치화 (1과 0으로 분류)

# 4. 훈련/검즈용 데이터 분리
from sklearn.model_selection import train_test_split

# 5. 분류모델구축
from sklearn.tree import DecisionTreeClassifier # 결정 트리
# from sklearn.naive_bayes import GaussianNB # 나이브 베이즈
# from sklearn.neighbors import KNeighborsClassifier # K-최근접 이웃
# from sklearn.ensemble import RandomForestClassifier # 랜덤 포레스트
# from sklearn.ensemble import BaggingClassifier # 앙상블
# from sklearn.linear_model import Perceptron # 페셉트론
# from sklearn.linear_model import LogisticRegression # 로지스틱 회귀 모델
# from sklearn.svm import SVC # 서포트 벡터 머신
# from sklearn.neural_network import MLPClassifier # 다층인공신경망

# 6.모델검정
from sklearn.metrics import confusion_matrix, classification_report # 정오분류표
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer # 정확도, 민감도 등
from sklearn.metrics import roc_curve # ROC 곡선 그리기

# 7.최적화
from sklearn.model_selection import cross_validate # 교차타당도
from sklearn.pipeline import make_pipeline # 파이프라인 구축
from sklearn.model_selection import learning_curve, validation_curve # 학습곡선, 검증곡선
from sklearn.model_selection import GridSearchCV # 9.하이퍼파라미터 튜닝

# (1) 데이터 가져오기

rm_df = pd.read_csv('RidingMowers.csv')
print(rm_df.head())

# 자료구조 살펴보기
print(rm_df.shape) 
# 행 / 렬 (48,3)

print(rm_df.keys()) # columns 의 index 값
# Index(['Income', 'Lot_Size', 'Ownership'], dtype='object')

# (2) data와 target으로 분리 (필요한 데이터만 추출) (data: X, target: y)

X = rm_df.drop(['Ownership'], axis=1)
print(X.head())

y = rm_df['Ownership']
print(y.head())

# (3) 데이터 전처리
# Class (target) 레이블 인코딩

class_le = LabelEncoder()
y = class_le.fit_transform(y)
print(y)

# (4) 훈련/검증용 데이터 분할
# \ 이후에 space 없어야 함
# test_size: 검증 데이터의 경우 30% 
# random_state: randoom seed 번호 1 
# stratify: y(클래스)의 비율에 따라 분할
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, 
                         test_size=0.3, 
                         random_state=1, 
                         stratify=y)

# (5) 모델 구축
tree = DecisionTreeClassifier(criterion = 'gini',
                              max_depth = 1,
                              random_state=1)
tree.fit(X_train, y_train)

# (6) 모델 검증
# tree_predict (class로 표시)
# tree_predict_proba (확률값으로 표시)
y_pred = tree.predict(X_test)
print(' 모델 검증: ', y_pred) # 추정한 값
print('실제 값:    ', y_test)

y_pred_p = tree.predict_proba(X_test)
print(' 모델 검증 확률값: ', y_pred_p) # 추정한 값

# 정오분류표로 모델 검증한다.
confmat = pd.DataFrame(confusion_matrix(y_test, y_pred),
                      index=['True[0]','True[1]'],
                      columns=['Predict[0]', 'Predict[1]'])
print(confmat)

print('Classification Report')
print(classification_report(y_test, y_pred))

# 정확도, 민감도 확인
print('잘못 분류된 샘플 개수: %d' % (y_test != y_pred).sum())
print('정확도: %.3f' % accuracy_score(y_test, y_pred))
print('정밀도: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('재현율: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

fpr, tpr, thresholds = roc_curve(y_test, tree.predict_proba(X_test)[:, 1])

print(fpr)
print(tpr)
print(thresholds)

plt.plot(fpr, tpr, '--', label="Logistic Regression")
plt.plot([0, 1], [0, 1], 'k--', label="random guess")
plt.plot([fpr], [tpr], 'r-', ms=10)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC')
plt.show()