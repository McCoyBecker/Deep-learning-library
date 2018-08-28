############################################################
# A feed forward neural network with a single dense layer. #
# Implementation using numpy. Applied to MNIST.            #
############################################################

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
from __classes.data_processing_funcs import batch_sample
from __classes.model import model
from __classes import optimizers as opt

# Load data into a training batch and a testing batch then normalize the data.
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train, x_test = x_train / 255, x_test / 255

# Define network parameters.
input_size = 784
num_classes = 10

# Set training parameters
epochs = 3
train_iterations = 200
batch_size = 100
initial_learning_rate = 0.1

## Construct the model ##

# First, setup the weights and biases using scipy truncnorm.

# truncnorm works by drawing from (a,b) interval over a scaled normal distribution.
# For consistency with the TensorFlow implementation, we should take a = -0.1, b = 0.1.
weights = {
    'weights_fc_1': truncnorm.rvs(-0.1, 0.1, size=(7*7*16,1024)),
    # output layer, 1024 inputs, 10 outputs (class prediction)
    'output': truncnorm.rvs(-0.1, 0.1, size=(1024,num_classes))
}
bias = {   
    'bias_fc_1': np.array([0.1 for i in range(1024)]),
    # Bias for the output layer
    'output': np.array([0.1 for i in range(num_classes)])
}

# Now, construct the layers.
dense_1 = layers.dense(weights['weights_fc_1'], bias['bias_fc_1'], ut.rectifier)
output = layers.dense(weights['output'], bias['output'], ut.softmax)

# Construct the model and add the layers.
pt2_model = model()
pt2_model.addLayer(dense_1)
pt2_model.addLayer(output)

# Construct the optimizer.
AdaGrad = opt.AdaGrad(pt2_model, ut.cross_entropy, initial_learning_rate)

# Construct an initial batch.
(batch, labels, batch_size) = batch_sample(x_train, y_train, batch_size)
AdaGrad.updateBatch(batch, labels, batch_size)
AdaGrad.init_gradient_memory()

# Implement training over epochs.
for epoch in range(epochs):
    
    # Initialize arrays to hold loss and batch accuracy.
    loss_function = batch_accuracy = np.zeros(train_iterations)

    # Reset the gradient memory.
    AdaGrad.init_gradient_memory()

    # Training iterations with AdaGrad.
    for i in range(train_iterations):
        print("\nThis is training iteration (" + str(i) +").")
        (loss_function[i], batch_accuracy[i]) = AdaGrad.optimize(initial_learning_rate)
        (batch, labels, batch_size) = batch_sample(x_train, y_train, batch_size)
        AdaGrad.updateBatch(batch, labels, batch_size)

# Generate predictions on the test set.
(x_test, y_test, test_size) = batch_sample(x_test, y_test, 10000)
prediction = pt2_model.feedforward(x_test)

# Initialize the test target array.
target = np.zeros((test_size, 10))
for i in range(test_size):
    target[i, int(y_test[i])] = 1

# Calculate the test accuracy and print.
test_accuracy = sum([1 for i in range(test_size) if np.argmax(prediction[i])  == np.argmax(target[i])])/test_size
print("The accuracy on the test set is: " + str(test_accuracy))