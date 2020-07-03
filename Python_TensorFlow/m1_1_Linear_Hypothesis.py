# 세상에는 Linear 한 형태로 설명될 수 있는 현상들이 많이 있다
# 예를 들어, 공부를 많이 하면 성적이 좋다든지, 평수가 넓으면 집의 가격이
# 비싼 경우를 들 수 있다.

# Linear Hypothesis: Data에 잘 맞는 직선을 찾는 과정이다. 
# H(x) = Wx + b 라고 가설을 세운다. W와 b에 따라 1차원 방정식의 직선의
# 형태가 달라진다.

# cost function: 거리를 측정하는 것이다. 즉, 1차원 방정식의 직선과 data를 비교
# H(x) - y  / (H(x) - y) ^ 2 이렇게 거리를 측정하기도 한다.
# 즉 cost function은 가장 작게 하는 W와 b를 찾는 문제이다. 얼마나 예측을 잘했는가를 나타내는 것이 cost function이다.

import tensorflow as tf

x_train = [1,2,3,4,5]
y_train = [1,2,3,4,5]
                            # 값이 1차원인 array이다.
W = tf.Variable(tf.random.normal([1]), name='Weight') # trainable variable이다. 학습을 위한 variable이다.
b = tf.Variable(tf.random.normal([1]), name='Bias')   # trainable variable이다. 학습을 위한 variable이다.
# tf.random.normal(
#    shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=None, name=None
#)

# Our hypothesis Wx + b
hypothesis = x_train * W + b
# Build hypothesis and cost
cost  = tf.reduce_mean(tf.square(hypothesis - y_train)) # reduce_mean: 평균내주는 함수이다.

# Minimize 
#optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
# learning rate initialize
learning_rate = 0.01

for i in range(2000):
    # Gradient Descent
    with tf.GradientTape() as tape:
        hypothesis = W * x_train + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_train))

    W_grad, b_grad = tape.gradient(cost, [W, b])

    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 20 == 0:
        #print("{:5}|{:10.4}|{:10.4}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))
        print(i)
        print(W)
        print(b)
        print(cost)

# Predict
print(W*5+b)
print(W*2.5+b)  