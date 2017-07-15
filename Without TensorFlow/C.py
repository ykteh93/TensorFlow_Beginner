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


# Relu function
def relu(x):
    return np.maximum(0,x)


# Draw random batch of training data
def random_batch(x, y, batchSize):
    idx = np.random.choice(np.arange(len(x)), len(x), replace=False)
    x   = x[idx[:batchSize]]
    y   = y[idx[:batchSize]]
    return x, y


# Forward propagation
def feed_forward(x,true_y):
    layer_1   = relu(x.dot(weights['hidden_1']) + biases['hidden_1'])
    layer_2   = relu(layer_1.dot(weights['hidden_2']) + biases['hidden_2'])
    predict_y = softmax(layer_2.dot(weights['out']) + biases['out'])
    accuracy  = (np.sum(np.argmax(predict_y, axis=1) == np.argmax(true_y, axis=1))) / len(true_y)
    loss      = cross_entropy_loss(true_y, predict_y)
    return layer_1, layer_2, predict_y, accuracy, loss


# Backward propagation
def feed_backward(layer_1, layer_2, predicted_y, x, true_y):
    delta             = (predicted_y - true_y) / len(true_y)
    dW                = (layer_2.T).dot(delta)
    dB                = np.sum(delta, axis=0, keepdims=True)

    dh2               = np.dot(delta, weights['out'].T)
    dh2[layer_2 <= 0] = 0
    dW_h2             = np.dot(layer_1.T, dh2)
    dB_h2             = np.sum(dh2, axis=0, keepdims=True)

    dh1               = np.dot(dh2, weights['hidden_2'].T)
    dh1[layer_1 <= 0] = 0
    dW_h1             = np.dot(x.T, dh1)
    dB_h1             = np.sum(dh1, axis=0, keepdims=True)

    return dW, dW_h2, dW_h1, dB, dB_h2, dB_h1


# Stochastic gradient descent
def gradient_descent(train_x, train_y, test_x, test_y):
    Train_error = Test_error = np.array([])
    # sys.stdout  = open('Train_and_Test_Error.txt', 'w')

    for epochs in range(number_epochs):
        for e in range(int(mnist.train.num_examples / batch_size)):
            # Draw random batch of 200 data set for training
            minibatch_x, minibatch_y = random_batch(train_x, train_y, 200)

            # Forward  and backward propagation
            layer_1, layer_2, predict_y, _, _  = feed_forward(minibatch_x, minibatch_y)
            dW, dW_h2, dW_h1, dB, dB_h2, dB_h1 = feed_backward(layer_1, layer_2, predict_y, minibatch_x, minibatch_y)

            # Update the weight and bias with gradient descent
            weights['hidden_1'] += - learning_rate * dW_h1
            weights['hidden_2'] += - learning_rate * dW_h2
            weights['out']      += - learning_rate * dW
            biases['hidden_1']  += - learning_rate * dB_h1
            biases['hidden_2']  += - learning_rate * dB_h2
            biases['out']       += - learning_rate * dB

        _, _, _, train_acc, train_loss            = feed_forward(train_x, train_y)
        _, _, predict_test_y, test_acc, test_loss = feed_forward(test_x, test_y)
        print("Epoch %4d -- Train Loss: %f -- Test Loss: %f -- Train Accuracy: %f -- Test Accuracy: %f" % (epochs, train_loss, test_loss, train_acc, test_acc))
        Train_error = np.append(Train_error, train_loss)
        Test_error  = np.append(Test_error, test_loss)

    # Plot train and test error over all epochs
    plot_train_test_error(Train_error, Test_error)

    # Draw confusion matrix
    confusion_matrix(predict_test_y, test_y)


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
learning_rate  = 0.01  # Learning rate for stochastic gradient descent
batch_size     = 100   # Size of mini batch for each training
number_epochs  = 100   # Number of desired epochs
features_1     = 256   # Number of features for the first hidden layer
features_2     = 256   # Number of features for the second hidden layer
number_input   = 784   # Data input (reshape from 28x28 to  784)
number_classes = 10    # Total number of classes (0-9 digits)

# Extract all the train and test set
train_x = mnist.train.images
train_y = mnist.train.labels
test_x  = mnist.test.images
test_y  = mnist.test.labels

# Define all the weights and biases
weights = {'hidden_1': np.random.standard_normal((number_input, features_1)),
           'hidden_2': np.random.standard_normal((features_1, features_2)),
           'out'     : np.random.standard_normal((features_2, number_classes))}

biases  = {'hidden_1': np.random.standard_normal((1,features_1)),
          'hidden_2' : np.random.standard_normal((1,features_2)),
          'out'      : np.random.standard_normal((1,number_classes))}

# Compute the weight and bias which reduces the cross entropy loss by gradient descent
gradient_descent(train_x, train_y, test_x, test_y)

# Store model parameter of weights and biases
np.savez('./model/weights.npz', out=weights['out'], hidden_1=weights['hidden_1'], hidden_2=weights['hidden_2'])
np.savez('./model/biases.npz', out=biases['out'], hidden_1=biases['hidden_1'], hidden_2=biases['hidden_2'])