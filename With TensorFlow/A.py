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
Train_error = Test_error = np.array([])
learning_rate = 0.8  # Learning rate for stochastic gradient descent
batch_size    = 100  # Size of mini batch for each training
number_epochs = 10   # Number of desired epochs

# Network Parameters
number_input   = 784 # Data input (reshape from 28x28 to 784)
number_classes = 10  # Total number of classes (0-9 digits)

# Graph Input
input_x = tf.placeholder(tf.float32, shape=[None, number_input])
true_y  = tf.placeholder(tf.float32, shape=[None, number_classes])

# All layers of weights and biases
weights = {'out': tf.Variable(tf.zeros([number_input, number_classes]))}
biases  = {'out': tf.Variable(tf.zeros([number_classes]))}

# Predict the label
predict_y = tf.matmul(input_x, weights['out']) + biases['out']

# Compute the cross-entropy loss and optimize with stochastic gradient descent to reduce the loss
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict_y, true_y))
training           = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_loss)

# Compute the average accuracy by counting the number of correct predicted label
accuracy         = tf.equal(tf.argmax(predict_y, 1), tf.argmax(true_y, 1))
average_accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

# Join the predicted label with the true label for the plot of confusion matrix
joint_labels = tf.pack([tf.argmax(predict_y,1), tf.argmax(true_y,1)])

# Initialize all variables and start the session
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    sys.stdout = open('Train_and_Test_Error.txt', 'w')

    for epochs in range(batch_size*number_epochs):
        minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)
        sess.run(training, feed_dict={input_x: minibatch_x, true_y: minibatch_y})

        # Display the training error of each mini batch and the test error of the whole test set at each epochs
        if epochs % batch_size == 0:
            error_mini_train = 1-sess.run(average_accuracy, feed_dict={input_x: minibatch_x, true_y: minibatch_y})
            error_full_train = 1-sess.run(average_accuracy, feed_dict={input_x: mnist.train.images, true_y: mnist.train.labels})
            error_test       = 1-sess.run(average_accuracy, feed_dict={input_x: mnist.test.images, true_y: mnist.test.labels})
            print("Epoch %4d -- Mini Batch Train Error: %f -- Full Set Train Error: %f -- Full Set Test Error: %f" % (epochs/ batch_size, error_mini_train, error_full_train, error_test))
            Train_error = np.append(Train_error, error_full_train)
            Test_error  = np.append(Test_error, error_test)

    # Plot the Error Rate of Train and Test Set
    x = np.linspace(0, number_epochs-1, num=number_epochs)
    plt.plot(x, Train_error, 'r', label='Train Error')
    plt.plot(x, Test_error, 'b', label='Test Error')
    plt.title('Plot of Train and Test Error Over %d Epochs' %(number_epochs))
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.savefig('Plot of Train and Test Error Over 100 Epochs.png')

    # Compute the accuracy and error rate for the whole set of train and test data
    accuracy_train = sess.run(average_accuracy, feed_dict={input_x: mnist.train.images, true_y: mnist.train.labels})
    accuracy_test  = sess.run(average_accuracy, feed_dict={input_x: mnist.test.images, true_y: mnist.test.labels})
    print("\nOverall Train Accuracy %g" % accuracy_train)
    print("Overall Test Accuracy %g" % accuracy_test)
    print("Overall Train Error Rate %g" % (1-accuracy_train))
    print("Overall Test Error Rate %g \n" % (1-accuracy_test))

    # Plot the confusion matrix
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
    saver.save(sess, './model/model.checkpoint')
