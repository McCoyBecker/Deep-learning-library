#################################################################
# A deep feedforward neural network with convolutional layers.  #
# Implementation using numpy. Applied to MNIST.                 #
#################################################################

# Import numpy. Import scipy for truncated normal distribution.
# Import TensorFlow for MNIST data set.
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
mnist = tf.keras.datasets.mnist

# Import utility functions, layers, optimizers, and the model.
from __classes import layers
from __classes import utility_funcs as ut
from __classes.regularizers import L2_reg
from __classes.data_processing_funcs import batch_sample
from __classes.model import model
from __classes import optimizers as opt

# Load data into a training batch and a testing batch then normalize the data.
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train, x_test = x_train / 255, x_test / 255

# Define class names.
class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

# Define network parameters.
input_size = 784
num_classes = 10

# Set training parameters
epochs = 10
train_iterations = 200
batch_size = 32
initial_learning_rate = 0.01

## ------ Construct the model ------ ##

# First, setup the weights and biases using scipy truncnorm.

# truncnorm works by drawing from (a,b) interval over a scaled normal distribution.
# For consistency with the TensorFlow implementation, we should take a = -0.1, b = 0.1.
weights = {
    
    # 10 by 10 filter convolution layer, 1 input, 32 outputs
    'weights_conv_1': truncnorm.rvs(-0.1, 0.1, size=(1,32,10,10)),
    # 5 by 5 filter convolution layer, 32 inputs, 16 outputs
    'weights_conv_2': truncnorm.rvs(-0.1, 0.1, size=(32,16,5,5)),
    # fully connected layer, 7*7*16 inputs, 1024 outputs
    'weights_fc_1': truncnorm.rvs(-0.1, 0.1, size=(7*7*16,1024)),
    # output layer, 1024 inputs, 10 outputs (class prediction)
    'output': truncnorm.rvs(-0.1, 0.1, size=(1024,num_classes))
}
bias = {   
    
    # Bias for convolution layer 1
    'bias_conv_1': np.array([0.1 for i in range(32)]),
    # Bias for convolution layer 2
    'bias_conv_2': np.array([0.1 for i in range(16)]),
    # Bias for the fully connected layer
    'bias_fc_1': np.array([0.1 for i in range(1024)]),
    # Bias for the output layer
    'output': np.array([0.1 for i in range(num_classes)])
}

# Now, construct the layers.
conv_layer_1 = layers.conv_2D(weights['weights_conv_1'], bias['bias_conv_1'], ut.leaky_rectifier)
max_pool_1 = layers.max_pool_2D(3, (32, 11, 11))
conv_layer_2 = layers.conv_2D(weights['weights_conv_2'], bias['bias_conv_2'], ut.leaky_rectifier)
max_pool_2 = layers.max_pool_2D(2, (16, 7, 7))
dense_1 = layers.dense(weights['weights_fc_1'], bias['bias_fc_1'], ut.leaky_rectifier)
output = layers.dense(weights['output'], bias['output'], ut.softmax)

# Construct the model and add the layers.
pt2_model = model()
pt2_model.addLayer(conv_layer_1)
pt2_model.addLayer(max_pool_1)
pt2_model.addLayer(conv_layer_2)
pt2_model.addLayer(max_pool_2)
pt2_model.addLayer(dense_1)
pt2_model.addLayer(output)

# Construct the optimizer.
AdaGrad = opt.AdaGrad(pt2_model, ut.cross_entropy, initial_learning_rate, L2_reg)

# Construct an initial batch.
(batch, labels, batch_size) = batch_sample(x_train, y_train, batch_size)
AdaGrad.updateBatch(batch, labels, batch_size)

# Initialize arrays to hold loss and batch accuracy.
batch_accuracy = np.zeros(train_iterations * epochs)
loss_function = np.zeros(train_iterations * epochs)

# Reset the gradient memory.
AdaGrad.init_gradient_memory()

# Implement training over epochs.
for epoch in range(epochs):

    # Training iterations with AdaGrad.
    for i in range(train_iterations):
        print("\nThis is training iteration (" + str(i) +").")
        (loss_function[i + train_iterations * epoch], batch_accuracy[i + train_iterations * epoch]) = AdaGrad.optimize(initial_learning_rate)
        (batch, labels, batch_size) = batch_sample(x_train, y_train, batch_size)
        AdaGrad.updateBatch(batch, labels, batch_size)


# Print the batch accuracy and average loss as a function of iteration step. 
plt.style.use('ggplot')
plt.subplot(211) 
plt.plot([i for i in range(train_iterations * epochs)], 1-batch_accuracy, c='b')
plt.title('Percent incorrect classification per batch')
plt.subplot(212)
plt.plot([i for i in range(train_iterations * epochs)], loss_function, c='r', label='Average loss function')
plt.title('Average loss function')
plt.xlabel('Iterations')
plt.show()

# Generate predictions on the test set.
(x_test, y_test, test_size) = batch_sample(x_test, y_test, 10000)
prediction = pt2_model.feedforward(x_test)

# Initialize the test target array.
target = np.zeros((test_size, 10))
for i in range(test_size):
    target[i, int(y_test[i])] = 1

# Calculate accuracy on the test set.
test_accuracy = sum([1 for i in range(test_size) if np.argmax(prediction[i])  == np.argmax(target[i])])/test_size
print("The accuracy on the test set is: " + str(test_accuracy))