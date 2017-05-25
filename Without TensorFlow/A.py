########################################################
#               Written by: Yih Kai Teh                #
########################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=np.nan)

# import the MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# Cross entropy loss function
def cross_entropy_loss(true_y, predict_y):
    return np.mean(-np.sum(np.multiply(true_y, np.log(predict_y)), 1))


# Softmax function
def softmax(x):
    x = x - np.max(x, axis=1).reshape((x.shape[0], 1))
    return np.exp(x)/np.sum(np.exp(x), axis=1).reshape((np.exp(x).shape[0], 1))


# Draw random batch of training data
def random_batch(x, y, batchSize):
    idx = np.random.choice(np.arange(len(x)), len(x), replace=False)
    x = x[idx[:batchSize]]
    y = y[idx[:batchSize]]
    return x, y


# Forward propagation
def feed_forward(x,true_y,weight,bias):
    predict_y = softmax(x.dot(weight) + bias)
    mistake = (np.sum(np.argmax(predict_y, axis=1) != np.argmax(true_y, axis=1))) / len(true_y)
    return predict_y, mistake


# Backward propagation
def feed_backward(predicted_y, x, true_y):
    delta = (predicted_y - true_y) / len(true_y)
    dW = (x.T).dot(delta)
    dB = np.sum(delta, axis=0, keepdims=True)
    return dW, dB


# Stochastic gradient descent
def gradient_descent(train_x, train_y, test_x, test_y):
    Train_error = np.array([])
    Test_error = np.array([])
    sys.stdout = open('Train_and_Test_Error.txt', 'w')

    for epochs in range(batch_size * number_epochs):
        # Draw random batch of 200 data set for training
        minibatch_x, minibatch_y = random_batch(train_x, train_y, 200)

        # Forward  and backward propagation
        predict_y, _  = feed_forward(minibatch_x, minibatch_y, weights['out'], biases['out'])
        dW, dB = feed_backward(predict_y, minibatch_x, minibatch_y)

        # Update the weight and bias with gradient descent
        weights['out'] += - learning_rate * dW
        biases['out'] += - learning_rate * dB

        if epochs % batch_size == 0:
            _, error_mini_train = feed_forward(minibatch_x, minibatch_y, weights['out'], biases['out'])
            _, error_full_train = feed_forward(minibatch_x, minibatch_y, weights['out'], biases['out'])
            _, error_test = feed_forward(test_x, test_y, weights['out'], biases['out'])
            print("Epoch %4d -- Mini Batch Train Error: %f -- Full Set Train Error: %f -- Full Set Test Error: %f" % (epochs/ batch_size, error_mini_train, error_full_train, error_test))
            Train_error = np.append(Train_error, error_full_train)
            Test_error = np.append(Test_error, error_test)

    # Plot train and test error over all epochs
    plot_train_test_error(Train_error, Test_error)

    # Compute the accuracy and error rate for the whole set of train and test data
    _, error_full_train = feed_forward(train_x, train_y, weights['out'], biases['out'])
    predict_test_y, error_test = feed_forward(test_x, test_y, weights['out'], biases['out'])
    print("\nOverall Train Accuracy %g" % (1 - error_full_train))
    print("Overall Test Accuracy %g" % (1 - error_test))
    print("Overall Train Error Rate %g" % error_full_train)
    print("Overall Test Error Rate %g \n" % error_test)

    # Draw confusion matrix
    confusion_matrix(predict_test_y, test_y)

    return weights['out'], biases['out']


# Plot the Error Rate of Train and Test Set
def plot_train_test_error(Train_error, Test_error):
    x = np.linspace(0, number_epochs-1, num=number_epochs)
    plt.plot(x, Train_error, 'r', label='Train Error')
    plt.plot(x, Test_error, 'b', label='Test Error')
    plt.title('Plot of Train and Test Error Over %d Epochs' %(number_epochs))
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.savefig('Plot of Train and Test Error Over 100 Epochs.png')

# Plot confusion matrix
def confusion_matrix(predict_y, true_y):
    total_list = np.asarray([np.argmax(predict_y, axis=1), np.argmax(true_y, axis=1)])
    confusion_matrix = np.zeros([10,10],int)
    for i in total_list.T:
        confusion_matrix[i[0],i[1]]+=1
    print('------------------ Confusion Matrix ------------------')
    print(pd.DataFrame(confusion_matrix))

# Parameters
learning_rate = 0.8  # Learning rate for stochastic gradient descent
batch_size = 100     # Size of mini batch for each training
number_epochs = 100  # Number of desired epochs
number_input = 784   # Data input (reshape from 28x28 to  784)
number_classes = 10  # Total number of classes (0-9 digits)

# Extract all the train and test set
train_x = mnist.train.images
train_y = mnist.train.labels
test_x = mnist.test.images
test_y = mnist.test.labels

# Define all the weights and biases
weights = {'out': np.zeros((number_input, number_classes))}
biases = {'out': np.zeros((1, number_classes))}

# Compute the weight and bias which reduces the cross entropy loss by gradient descent
weights['out'], biases['out'] = gradient_descent(train_x, train_y, test_x, test_y)

# Store model parameter of weights and biases
np.savez('./model/weights.npz', out=weights['out'])
np.savez('./model/biases.npz', out=biases['out'])

# Store the parameter value of weight and bias into text file
sys.stdout = open('Parameter_P2_b (All Weights & Biases).txt', 'w')
print('------------------ Weights ------------------')
print(weights['out'])
print('\n------------------ Biases ------------------')
print(biases['out'])