########################################################
#               Written by: Yih Kai Teh                #
########################################################

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=np.nan)

# import the MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Parameters
learning_rate  = 0.8  # Learning rate for stochastic gradient descent
batch_size     = 100  # Size of mini batch for each training
number_epochs  = 10   # Number of desired epochs
number_input   = 784  # Data input (reshape from 28x28 to 784)
number_classes = 10   # Total number of classes (0-9 digits)
Train_error    = Test_error = np.array([]) # for graph plotting

# Graph Input
input_x = tf.placeholder(tf.float32, shape=[None, number_input])
true_y  = tf.placeholder(tf.float32, shape=[None, number_classes])

# All layers of weights and biases
weights = {'out': tf.Variable(tf.zeros([number_input, number_classes]))}
biases  = {'out': tf.Variable(tf.zeros([number_classes]))}

# Predict the label
predict_y = tf.matmul(input_x, weights['out']) + biases['out']

# Compute the cross-entropy loss and optimize with stochastic gradient descent to reduce the loss
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict_y, labels = true_y))
training           = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_loss)

# Compute the average accuracy by counting the number of correct predicted label
accuracy           = tf.equal(tf.argmax(predict_y, 1), tf.argmax(true_y, 1))
average_accuracy   = tf.reduce_mean(tf.cast(accuracy, tf.float32))

# Join the predicted label with the true label for the plot of confusion matrix
joint_labels       = tf.stack([tf.argmax(predict_y,1), tf.argmax(true_y,1)])

# Initialize all variables and start the session
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    saver      = tf.train.Saver()
    sys.stdout = open('Train_and_Test_Error.txt', 'w')

    for epochs in range(number_epochs):
        for e in range(int(mnist.train.num_examples / batch_size)):
            minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)
            sess.run(training, feed_dict={input_x: minibatch_x, true_y: minibatch_y})

        # Display the training error of each mini batch and the test error of the whole test set at each epochs
        train_loss, train_acc = sess.run([cross_entropy_loss, average_accuracy], feed_dict={input_x: mnist.train.images, true_y: mnist.train.labels})
        test_loss,  test_acc  = sess.run([cross_entropy_loss, average_accuracy], feed_dict={input_x: mnist.test.images,  true_y: mnist.test.labels})
        print("Epoch %4d -- Train Error: %f -- Test Error: %f -- Train Accuracy: %f -- Test Accuracy: %f" % (epochs, train_loss, test_loss, train_acc, test_acc))
        Train_error = np.append(Train_error, train_loss)
        Test_error  = np.append(Test_error, test_loss)

    # Plot the Error Rate of Train and Test Set
    x = np.linspace(0, number_epochs-1, num=number_epochs)
    plt.plot(x, Train_error, 'r', label='Train Error')
    plt.plot(x, Test_error, 'b', label='Test Error')
    plt.title('Plot of Train and Test Error Over %d Epochs' %(number_epochs))
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.savefig('Plot of Train and Test Error Over 100 Epochs.png')

    #Plot the confusion matrix
    total_list = sess.run(joint_labels, feed_dict={input_x: mnist.test.images, true_y: mnist.test.labels})
    confusion_matrix = np.zeros([10,10],int)
    for i in total_list.T:
        confusion_matrix[i[0],i[1]]+=1
    print('------------------ Confusion Matrix ------------------')
    print(pd.DataFrame(confusion_matrix))

    # Store the parameter value of weight and bias into text file
    sys.stdout = open('Parameter_P1_a (All Weights & Biases).txt', 'w')
    print('------------------ Weights ------------------')
    print(weights['out'].eval())
    print('\n------------------ Biases ------------------')
    print(biases['out'].eval())

    # Save the model
    saver.save(sess, './model/A')