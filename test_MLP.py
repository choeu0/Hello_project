#버전 오류 때문에
import warnings
warnings.filterwarnings('ignore')


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import random

#버전 오류 때문에
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)



 
# loading data
# read_data_sets: 간편하게 데이터를 객체 형채로 받아옴
# one_hot: 인코딩된 형태로 받아옴
mnist = input_data.read_data_sets("./MNIST_DATA", one_hot=True)


# MLP
# 784: 28*28=784 -> 벡터로 표현
# None: 이미지가 몇개 들어가느냐
# placeholder: 레이어에 들어가는 노드 생성시, 현재 값은 없고 자료형만 들어가있음
X = tf.placeholder(tf.float32,[None,784]) 



# weight, bias 값 초기화 / 정규분포 랜덤 값
# 784: 들어가는 값, 10: 나오는 값 0~9
W = tf.Variable(tf.random_normal([784, 10], stddev=0.1))
b = tf.Variable(tf.random_normal([10], stddev=0.1))


# using softmax classifier
# model
# matmul: 행렬을 곱해주는 함수
# softmax: 모든 확률의 합이 1이 되도록 하는 함수
# 숫자 -> softmax 가능
y = tf.nn.softmax(tf.matmul(X, W) + b)

#cross_entropy
Y = tf.placeholder(tf.float32,[None,10])
 
# loss: 모델이 원하는 결과에서 얼마나 떨어져 있나 보여줌, 작을수록 좋음
# cross entroy
# 경사하강법: loss를 최소화하는 방향
loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

 
# Test model
# 예측 결과값: tf.arg_max(y,1), 실제 결과값: tf.arg_max(Y, 1)
# correct: 부울값
correct = tf.equal(tf.arg_max(y,1), tf.arg_max(Y, 1))

# Calculate accuracy
# tf.cast: 형변환
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
 
# parameters
# Training epoch/batch
# epoch: 전체 데이터를 한번 도는 것
# batch_size: 데이터를 쪼개서
epochs = 15
batch_size = 100

# Init Tensorflow variables
init =tf.global_variables_initializer()

sess = tf.Session()    
sess.run(init)



# Training cycle
for epoch in range(epochs):
    avg_loss = 0
    total_batch = int(mnist.train.num_examples / batch_size)
 
    for i in range(batch_size):
        # 결과값으로 train_acc, loss
        # c, _: loss 결과값, train_acc 받아들이지 않음
        train_cost, train_acc = mnist.train.next_batch(batch_size)
        c, _= sess.run([loss,optimizer], feed_dict={X: train_cost, Y: train_acc})
        avg_loss += c / total_batch
 
    print('Epoch:', '%02d' %(epoch+1), 'loss = ', '{:.9f}'.format(avg_loss))

 
print("\nLearning finished")
# eval: 문자열로 입력된 값을 숫자형으로 바꿔줌
print("Accuracy: ", accuracy.eval(session=sess,feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
 
# Get one and predict using matplotlib
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(tf.argmax(y, 1), feed_dict={X: mnist.test.images[r:r + 1]}))
 
plt.imshow(
    mnist.test.images[r:r + 1].reshape(28, 28),
    cmap='Greys',
    interpolation='nearest')

plt.show()
 
