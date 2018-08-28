import numpy as np

# Define the softmax function. Input: ndarray. Output: ndarray.
def softmax(inputData):

    # Shift input data by max value to help prevent NaNs.
    inputData = inputData - max(inputData)

    # Initialize partition function for data.
    PartitionFunction = np.sum(np.exp(inputData))

    # Generate output.
    softmax = np.exp(inputData)/PartitionFunction

    # Return output.
    return(softmax)

# Define the derivative of the softmax function. Input: ndarray. Output: ndarray x ndarray.
def d_softmax(inputData):

    # Get input dimensions.
    width = inputData.shape[0]

    # Initialize derivative matrix.
    d_softmax = np.zeros((width, width))

    # Construct softmax array.
    softmax_array = softmax(inputData)

    # Construct the derivative matrix.
    for i in range(width):
        for j in range(width):
            if i == j:
                d_softmax[i,j] = softmax_array[i]*(1-softmax_array[j])
        
            else:
                d_softmax[i,j] = -softmax_array[i]*softmax_array[j]

    # Return the derivative matrix.
    return(d_softmax)

# Define the non-vectorized rectifier function. Input: num. Output: num. 
def nonvec_rectifier(inputData):

    # If the input data is greater than 0, return input data. Else return 0.
    if inputData > 0:
        return(inputData)
    else:
        return(0)

# Define a vectorized version of the rectifier. Input: ndarray. Output: ndarray.
def rectifier(inputData):

    # Convert non-vectorized version to vectorized.
    vec = np.vectorize(nonvec_rectifier)

    # Apply to input.
    output = vec(inputData)

    # Return output.
    return(output)

# Define the non-vectorized derivative of the rectifier. Input: num. Output: num.
def nonvec_d_rectifier(inputData):

    # If input data is greater than 0, return 1. Else return 0.
    if inputData > 0:
        return(1)
    else:
        return(0)

# Define the vectorized derivative of the rectifier. Input: ndarray. Output: ndarray.
def d_rectifier(inputData):

    # Convert non-vectorized version to vectorized.
    vec = np.vectorize(nonvec_d_rectifier)

    # Apply to input.
    output = vec(inputData)

    # Return output.
    return(output)

# Define non-vectorized leaky rectifier. Input: ndarray. Output: ndarray.
def nonvec_leaky_rectifier(inputData, leak = 0.001):

    # If inputData > 0, return 1. Else return input data multiplied by a leakiness coefficient.
    if inputData > 0:
        return(inputData)
    else:
        return(leak*inputData)

# Define vectorized leaky rectifier. Input: ndarray. Output: ndarray.
def leaky_rectifier(inputData, leak = 0.001):

    # Convert non-vectorized to vectorized.
    vec = np.vectorize(nonvec_leaky_rectifier)

    # Return output.
    return(vec(inputData, leak))

# Define non-vectorized derivative of the leaky rectifier. Input: ndarray. Output: ndarray.
def nonvec_d_leaky_rectifier(inputData, leak = 0.001):

    # If input data is greater than 0, return 1. Else return leak coefficient.
    if inputData > 0:
        return(1)
    else:
        return(leak)

# Define the vectorized derivative of the leaky rectifier. Input: ndarray. Output: ndarray.
def d_leaky_rectifier(inputData, leak = 0.001):

    # Convert non-vectorized to vectorized.
    vec = np.vectorize(nonvec_d_leaky_rectifier)

    # Return output.
    return(vec(inputData, leak))
    
# Define the cross-entropy. Input: ndarray x ndarray. Output = scalar.
def cross_entropy(predicted, target):

    # Convert non-vectorized to vectorized.
    log = np.vectorize(np.log)

    # Compute cross-entropy and return.
    return(-np.sum(np.multiply(target,log(predicted)), axis = 1))

# Define the derivative of cross-entropy. Input: ndarray x ndarray. Output = ndarray.
def d_cross_entropy(predicted, target):

    # Construct derivative.
    d_cross_entropy = np.multiply(-target, 1/predicted)

    # Return derivative.
    return(d_cross_entropy)