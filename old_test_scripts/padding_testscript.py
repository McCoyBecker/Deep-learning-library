from __classes.data_processing_funcs import pad_with
import numpy as np

# ---- A test script for padding in convolution and max pooling layers. ----

inputData = np.ones((1,1,4,4))
pad_array = np.zeros((1, 1, 4 + 2 * 1, 4 + 2 * 1))
for samp in range(1):
    for num in range(1):
        pad_array[samp, num] = np.pad(inputData[samp, num], 1, pad_with)
inputData = pad_array
print(inputData[0:, 0:, 0 + 1: 6 - 1, 0 + 1: 6-1])