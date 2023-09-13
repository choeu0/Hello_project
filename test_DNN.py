import warnings
warnings.filterwarnings('ignore')


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import random


old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

 
# loading data
mnist = input_data.read_data_sets("./MNIST_DATA", one_hot=True)


# DNN
X = tf.placeholder(tf.float32,[None,784]) 

 

# weight, bias 값 초기화 / 정규분포 랜덤 값
# 히든 레이어 3개
# [784,256]와 [256] 곱하려면 256으로 같아야 함
W1 = tf.Variable(tf.random_normal([784,256], stddev=0.1))
b1 = tf.Variable(tf.random_normal([256], stddev=0.1))
L1 = tf.nn.leaky_relu(tf.matmul(X, W1) + b1)
 
W2 = tf.Variable(tf.random_normal([256,256], stddev=0.1))
b2 = tf.Variable(tf.random_normal([256], stddev=0.1))
L2 = tf.nn.leaky_relu(tf.matmul(L1, W2) + b2)
 
# [256,10], [10] 마지막에는 10으로 <- 값이 0~9로 10개
W3 = tf.Variable(tf.random_normal([256,10], stddev=0.1))
b3 = tf.Variable(tf.random_normal([10], stddev=0.1))


# using softmax classifier
# model
y = tf.matmul(L2, W3) + b3


#cross_entropy
Y = tf.placeholder(tf.float32,[None,10])

# 벡터 -> softmax 여기에서
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

 
# Test model
correct = tf.equal(tf.arg_max(y,1), tf.arg_max(Y, 1))

# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
 
# parameters
# Training epoch/batch
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
        train_cost, train_acc = mnist.train.next_batch(batch_size)
        c, _= sess.run([loss,optimizer], feed_dict={X: train_cost, Y: train_acc})
        avg_loss += c / total_batch
 
    print('Epoch:', '%02d' %(epoch+1), 'loss = ', '{:.9f}'.format(avg_loss))

 
print("\nLearning finished")
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
 
