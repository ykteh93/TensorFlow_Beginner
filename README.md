# TensorFlow_Beginner
Classifying handwritten digit with MNIST dataset

This purpose of this project is to understand the foundation of forward and backward propagation of neural network.

<br><br/>
There are two versions (each with 3 files) in this repository:

<dl>
  <dt> 1. With TensorFlow implementation:</dt>
  <dd> This is similar to the tutorial provided at the official website of TensorFlow.</dd>
  
   <dt> 2. Without Tensorflow implementation:</dt>
  <dd> This is the exact equivalent but using only numpy to implement everything including chain-rule derivative in backward propagation.</dd>
</dl>

<dl>
  <dt> The architecture of neural network in each file:</dt>
  <dd> A: input &rarr; linear layer &rarr; softmax &rarr; class probabilities</dd>
  <dd> B: input &rarr; hidden layer (128 units) + Relu &rarr; linear layer &rarr; softmax &rarr; class probabilities</dd>
  <dd> C: input &rarr; 2 * hidden layer (256 units) + Relu &rarr; linear layer &rarr; softmax &rarr; class probabilities</dd>
</dl>

<br><br/>

The required library:
* TensorFlow
* numpy
* pandas
* matplotlib
