# Ch03. 기계학습 기초 ( Packages )
# 1. 기계학습이란
# 2. 기계학습 분석절차
# 3. 기본 데이터 분석

# 기계학습 (machine learning) : 데이터로부터 정확하게 예측할 수 있는지에 중점을 둔다. 정확하게 맞추는 것에 의미를 둔다.
# 데이터 / 해답을 넣고 learning 하면 규칙을 얻을 수 있다. 이것을 귀낙접 방법이라고 한다.

# Artificial Intelligence(기계를 지능적으로 만드는 것) 가 지각, 인지도 할 수 있게 됨 - 이미지 인지
# Machine Learning: 프로그래밍하지 않고 컴퓨터에게 학습할 수 있는 능력을 주는 것
# Deep Learning: 컴퓨터가 스스로 학습할 수 있게 하기 위해서 심층 신경망 (Deep Neural Network)으로 이루어지는 기계 학습

# Supervised Learning: 정답을 알 수 있어서 바로 바로 피드백을 받으면서 학습 (Classification)
# 지도학습의 목적: 분류와 예측
# Unsupervised Learning: 정답이 없는 분류와 같은 문제를 푸는 것 (연관성을 찾아 내는 것)
# 비지도학습: 데이터가 어떻게 구성되었는지를 알아내는 문제
# Reinforcement Learning: 정답은 모르지만, 자신이 한 행동에 대한 보상을 알 수 있어서 그로부터 학습하는 것 (알파고)
# 강화학습: 잘하면 보상을 주고 못하면 패널티를 주어 잘 하도록 유도하는 것

# Colaboratory 구글에서 만든 것! Big data 연습할 때 사용해라.

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
from sklearn.naive_bayes import GaussianNB # 나이브 베이즈
from sklearn.neighbors import KNeighborsClassifier # K-최근접 이웃
from sklearn.ensemble import RandomForestClassifier # 랜덤 포레스트
from sklearn.ensemble import BaggingClassifier # 앙상블
from sklearn.linear_model import Perceptron # 페셉트론
from sklearn.linear_model import LogisticRegression # 로지스틱 회귀 모델
from sklearn.svm import SVC # 서포트 벡터 머신
from sklearn.neural_network import MLPClassifier # 다층인공신경망

# 6.모델검정
from sklearn.metrics import confusion_matrix, classification_report # 정오분류표
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer # 정확도, 민감도 등
from sklearn.metrics import roc_curve # ROC 곡선 그리기

# 7.최적화
from sklearn.model_selection import cross_validate # 교차타당도
from sklearn.pipeline import make_pipeline # 파이프라인 구축
from sklearn.model_selection import learning_curve, validation_curve # 학습곡선, 검증곡선
from sklearn.model_selection import GridSearchCV # 9.하이퍼파라미터 튜닝
