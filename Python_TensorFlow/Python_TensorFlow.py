import tensorflow as tf
 
# cudnn64_7.dll 
# I added above file.

# 아래는 tf 1 버전  텐서플로우 버전 2.0.0에서는 Session을 정의하고 run 해주는 과정이 생략된다.

#welcome =tf.constant("Welcome to TensorFlow!!")
#sess = tf.Session()
#sess.run(welcome)

# 아래는 tf 2 버전 확인
print(tf.__version__) # 2.2.0

welcome = tf.constant("Welcome to TensorFlow!!")
tf.print(welcome)

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
mul = node1 * node2

tf.print(node1, node2)
tf.print(mul)