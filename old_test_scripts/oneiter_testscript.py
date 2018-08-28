# A test script for diagnosis and evaluation.

from scipy.stats import truncnorm
import numpy as np
import tensorflow as tf
mnist = tf.keras.datasets.mnist

from __classes import layers
from __classes import utility_funcs as ut
from __classes.data_processing_funcs import batch_sample
from __classes.model import model
from __classes import optimizers as opt

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

my_conv1 = layers.conv_2D(truncnorm.rvs(-0.2, 0.2, size=(1,32,10,10)), np.array([0.1 for i in range(32)]), ut.rectifier)
my_pool1 = layers.max_pool_2D(3, (32, 11, 11))
my_conv2 = layers.conv_2D(truncnorm.rvs(-0.2, 0.2, size=(32,16,5,5)), np.array([0.1 for i in range(16)]), ut.rectifier)
my_pool2 = layers.max_pool_2D(2, (16, 7, 7))
my_dense1 = layers.dense(np.array(truncnorm.rvs(-0.2, 0.2, size=(7*7*16,1024))), np.array([0.1 for i in range(1024)]), ut.rectifier)
my_output = layers.dense(np.array(truncnorm.rvs(-0.2, 0.2, size=(1024, 10))), np.array([0.1 for i in range(10)]), ut.softmax)

my_model = model()
pt2_model = model()
pt2_model.addLayer(my_conv1)
pt2_model.addLayer(my_pool1)
pt2_model.addLayer(my_conv2)
pt2_model.addLayer(my_pool2)
pt2_model.addLayer(my_dense1)
pt2_model.addLayer(my_output)

AdaGrad = opt.AdaGrad(pt2_model, ut.cross_entropy, 0.1)
(batch, labels, batch_size) = batch_sample(x_train, y_train, 10)
AdaGrad.updateBatch(batch, labels, batch_size)
AdaGrad.optimize()