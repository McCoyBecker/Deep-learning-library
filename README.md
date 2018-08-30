## A deep learning library implemented using NumPy

This is a NumPy-based library used to implement a deep convolutional neural network. The network is applied to the MNIST data set, which is acquired using TensorFlow/Keras.

The __classes directory contains utility functions (rectifier, leaky rectifier, softmax, and cross entropy), layers (conv_2d, dense, max pool), optimizers (AdaGrad), regularizers (L2), and the model class.

## IMPORTANT: 

When running numpy_convolution.py, I have set the formerly ReLu activation to leaky-ReLu to prevent too much hassle in tuning learning parameters.

## Future commits:

0. For convolution networks, a small AdaGrad gradient memory initialization tends to cause significant ReLu death. The rate at which exploding gradients appear is batch-size dependent as well. Exploding gradients return NaNs, which are set to 0 using NumPy's nan_to_num(). With a fixed learning rate of 0.01, some care must be taken to select an appropriate initialization. Possible solutions: implement leaky ReLu's, or return large negative numbers when NaNs occur in the average cross-entropy loss.

1. The library does not currently support convolutional layers without max pooling (max pooling is required to reshape backpropagated errors out of dense layers). For future commits, wrap reshape into utility function.

2. Implement an im2col or GEMM inspired way to do forward propagation in max pooling. This is currently the most inefficient task in the network.

3. Write the summary() function in model to spit out Keras-style summary of trainable parameters.

4. Add support for other optimizers.
