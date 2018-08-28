import numpy as np
from __classes.utility_funcs import rectifier, leaky_rectifier
from __classes.utility_funcs import softmax
from __classes.utility_funcs import d_softmax, d_cross_entropy, d_rectifier, d_leaky_rectifier
from __classes.data_processing_funcs import pad_with

# Fully-connected layer.
class dense:

    def __init__(self, weights, bias, activationFunction):
        self.weights = weights
        self.bias = bias
        self.activation = activationFunction
        self.activation_deriv = eval("d_" + activationFunction.__name__)

    # Forward propagation of information.
    def propagate(self, inputData):

        # Print input data shape.
        print("Input shape: " + str(inputData.shape))

        # Retrieve shape of input data. If data is not flattened, expect 4 and flatten. If flattened, expect 2.
        try:
            (num_samples, num_input, height, width) = inputData.shape

            # If input is multi-dimensional, flatten.
            flat_array = np.zeros((num_samples, num_input * height * width))
            for samp in range(num_samples):
                if inputData[samp].ndim != 1:
                    flat_array[samp] = inputData[samp].flatten()
            inputData = flat_array

            # Retrieve and print flattened data shape.
            (num_samples, width) = inputData.shape
            print("Flatten input shape: " + str(inputData.shape))

        except ValueError:
            (num_samples, width) = inputData.shape

        # Retrieve shape of weights. 
        (num_input, num_out) = self.weights.shape

        # Calculate activation input.
        activations = np.zeros((num_samples, num_out))
        for samp in range(num_samples):
            activations[samp] = np.matmul(inputData[samp], self.weights) + self.bias

        # Prepare output.
        output = np.zeros(activations.shape)

        # Calculate output.
        for samp in range(num_samples):
            output[samp] = self.activation(activations[samp])

        # Save input and activations for backpropagation.
        self.memory = (inputData, activations)

        # Print output shape.
        print("Output shape: " + str(output.shape))

        # Return the output.
        return(output)

    # Ye old backpropagation.
    def backpropagate(self, input_errors):

        # Print input errors shape.
        print("Input errors shape: " + str(input_errors.shape))

        # Recall input and activations.
        (input_memory, activations) = self.memory

        # Recover input dimensions.
        (num_samples, _) = input_memory.shape
        print("Remember, the original input had shape: " + str(input_memory.shape))
        print("The activations had shape: " + str(activations.shape))

        # Initialize output errors using shape of inputs.
        output_errors = np.zeros(input_memory.shape)

        # Multiply input errors by activation derivative. If the activation is softmax, we must matrix multiply.
        if self.activation_deriv.__name__ == 'd_softmax':
            for samp in range(num_samples):
                input_errors[samp] = np.matmul(input_errors[samp], self.activation_deriv(activations[samp]))

        # If the activation is other, we must Hadamard product.
        else:
            for samp in range(num_samples):
                input_errors[samp] = np.multiply(input_errors[samp], self.activation_deriv(activations[samp]))

        # Initialize weight and bias gradients.
        weight_gradients = np.zeros((num_samples,) + self.weights.shape)
        bias_gradients = np.zeros((num_samples,) + self.bias.shape)
        for samp in range(num_samples):
            weight_gradients[samp] = np.outer(input_memory[samp], input_errors[samp])
            bias_gradients[samp] += input_errors[samp]

        # Multiply input errors by weights and send out.
        for samp in range(num_samples):
            output_errors[samp] = np.matmul(input_errors[samp], np.transpose(self.weights))

        # Print output errors shape.
        print("Output errors shape: " + str(output_errors.shape))

        # Return output errors and gradients.
        return(output_errors, weight_gradients, bias_gradients)

    # Update the weights and biases.
    def updateWeights(self, weight_gradients, bias_gradient, weight_learning_rate, bias_learning_rate):
        self.weights = self.weights - np.multiply(weight_learning_rate, np.average(weight_gradients, axis = 0))
        self.bias = self.bias - np.multiply(bias_learning_rate, np.average(bias_gradient, axis = 0))

# 2D convolution layer.
class conv_2D:

    def __init__(self, weights, bias, activationFunction):
        self.weights = weights
        self.bias = bias
        self.activation = activationFunction
        self.activation_deriv = eval("d_" + activationFunction.__name__)

    # Forward propagation of information.
    def propagate(self, inputData):

        # Print input data shape.
        print("Input shape: " + str(inputData.shape))

        # Retrieve shape of input data.
        (num_samples, _, height, width) = inputData.shape

        # Retrieve filter shape from weights.
        (_, num_output, filter_dim, filter_dim) = self.weights.shape
        print("The shape of the weights is: " + str(self.weights.shape))

        # Determine output shape.
        output_height = height - filter_dim + 1
        output_width = width - filter_dim + 1

        # Initialize output.
        output = np.zeros((num_samples, num_output, output_height, output_width))

        # Apply convolution filters using NumPy's tensordot.
        for height in range(output_height):
            for width in range(output_width):
                input_slice = inputData[0:, 0:, height : height + filter_dim, width : width + filter_dim]
                convo_filter = np.rot90(np.rot90(self.weights[0:, 0:]))
                output[0:, 0:, height, width] = np.tensordot(input_slice, convo_filter, axes = ([1,2,3], [0,2,3]))

        for num in range(num_output):
            output[0:, num] += self.bias[num]

        # Save activation input data and input data for backpropagation.
        self.memory = (output, inputData)

        # Apply vectorized activation function.
        output = self.activation(output)

        # Print output shape.
        print("Output shape: " + str(output.shape))

        # Return output.
        return(output)

    # Ye old backpropagation.
    def backpropagate(self, input_errors): 

        # Print input errors shape.
        print("Input errors shape: " + str(input_errors.shape))

        # Recall old input to layer and activation.
        (activation_input, old_input) = self.memory
        print("Remember, the old input had shape: " + str(old_input.shape))
        print("Remember, the activations had shape: " + str(activation_input.shape))

        # Retrieve shape of weights.
        (_, _, filter_dim, filter_dim) = self.weights.shape
        print("The weights have shape: " + str(self.weights.shape))

        # Retrieve shape of old input.
        (num_samples, _, old_input_height, old_input_width) = old_input.shape

        # Retrieve shape of input errors.
        (num_samples, _, error_filter_dim, error_filter_dim) = np.transpose(input_errors).shape

        # Determine output shape.
        output_height = old_input_height - error_filter_dim + 1
        output_width = old_input_width - error_filter_dim + 1

        # Initialize weight and bias gradients.
        weight_gradients = np.zeros((num_samples,) + self.weights.shape)
        bias_gradients = np.zeros((num_samples,) + self.bias.shape)

        # Multiply input errors by activation derivative.
        if self.activation_deriv.__name__ == 'd_softmax':
            input_errors = np.matmul(input_errors, self.activation_deriv(activation_input))
        else:
            input_errors = np.multiply(input_errors, np.transpose(self.activation_deriv(activation_input)))

        # Initialize output error array.
        output_errors = np.zeros(old_input.shape)

        # -------- INEFFICIENT -------- #
        
        # Get the input errors before we transform them.
        error_slice = np.rot90(np.rot90(np.transpose(input_errors)))

        # To prepare the output errors, we apply padding around each input error sample.
        pad_coordinates = ((0,0), (0,0), (filter_dim-1, filter_dim-1), (filter_dim-1, filter_dim-1))
        input_errors = np.pad(np.transpose(input_errors), pad_width = pad_coordinates, mode = 'constant', constant_values = 0)

        # Now, we produce the next set of weight and bias gradients, and output errors.
        for height in range(output_height):
            for width in range(output_width):

                # Here's the weight and bias gradients.
                old_input_slice = old_input[0:, 0:, height : height + error_filter_dim, width : width + error_filter_dim]
                weight_gradients[0:, 0:, 0:, height, width] = np.sum(np.rot90(np.rot90(np.tensordot(old_input_slice, error_slice, axes = ([(2,3), (2,3)])))), axis = 2)
                bias_gradients[0:, 0:] = np.sum(error_slice)

                # Now, we produce the output errors.
                reshaped_error_slice = input_errors[0:, 0:, height : height + filter_dim, width : width + filter_dim]
                filter_slice = np.rot90(np.rot90(self.weights))
                output_errors[0:, 0:, height, width] = np.tensordot(reshaped_error_slice, filter_slice, axes = ([(1,2,3),(1,2,3)]))

        # -------- END -------- #

        # Print output shape.
        print("Output errors shape: " + str(output_errors.shape))

        # Return output errors and gradients.
        return(output_errors, weight_gradients, bias_gradients)

    # Update the weights and biases.
    def updateWeights(self, weight_gradients, bias_gradient, weight_learning_rate, bias_learning_rate):
        self.weights = self.weights - np.multiply(weight_learning_rate, np.average(weight_gradients, axis = 0))
        self.bias = self.bias - np.multiply(bias_learning_rate, np.average(bias_gradient, axis = 0))

# Max pooling layer.
class max_pool_2D:

    def __init__(self, padding, output_shape):
        self.stride = 2
        self.padding = padding
        self.output_shape = output_shape

    # Forward propagation of information.
    def propagate(self, inputData):

        # Print input data shape.
        print("Input shape: " + str(inputData.shape))

        # Retrieve shape of input data.
        (num_samples, num_input, height, width) = inputData.shape

        # If input data has odd dimensions, we pad a new column and row with -inf.
        if (height % 2 == 1 or width % 2 == 1):
            pad_array = np.full((num_samples, num_input, height + 1, width + 1), -np.inf)
            pad_array[0:, 0:, 0 : height, 0 : width] = inputData
            inputData = pad_array

        # Now we apply the max pool padding parameter around each input sample.
        pad_coordinates = ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding))
        inputData = np.pad(inputData, pad_width = pad_coordinates, mode = 'constant', constant_values = 0)

        # Get new shape.
        (num_samples, num_input, height, width) = inputData.shape

        # Save shape in memory for backpropagation.
        input_memory = inputData.shape

        # Retrieve shape of filter. Number of input and output nodes is conserved.
        (filter_dim, filter_dim) = (2,2)
        num_output = num_input

        # Determine output shape.
        output_height = height // self.stride
        output_width = width // self.stride

        # Initialize output.
        output = np.zeros((num_samples, num_output, output_height, output_width))

        # Initialize an array to store the indices of the maximum values in the input layer.
        max_indices = np.zeros(input_memory)

        # Apply the filter. 
        # ------ INEFFICIENT ------ #
        for samp in range(num_samples):
            for num in range(num_output):
                for height in range(output_height):
                    for width in range(output_width):
                        input_slice = inputData[samp, num, height*self.stride : height*self.stride + filter_dim, 
                                                width*self.stride : width*self.stride + filter_dim]
                        max_index = np.unravel_index(np.argmax(input_slice, axis = None), input_slice.shape)
                        max_indices[samp, num, max_index[0] + height, max_index[1] + width] = 1
                        output[samp, num, height, width] = np.amax(input_slice)
        # ------ END ------ #

        # Print output shape.
        print("Output shape: " + str(output.shape))

        # Store max indices and input in memory.
        self.memory = (max_indices, input_memory)

        # Return output.
        return(output)

    # Ye old backpropagation.
    def backpropagate(self, input_errors):

        # Print input errors shape.
        print("Input errors shape: " + str(input_errors.shape))

        # Check input dimension, then reshape.
        try:
            (num_samples, _) = input_errors.shape
            reshape_arr = np.zeros((num_samples,) + self.output_shape)
            for samp in range(num_samples):
                reshape_arr[samp] = np.reshape(input_errors[samp], self.output_shape)
            input_errors = reshape_arr
            print("The input has been reshaped. The shape is: " + str(input_errors.shape))
        
        except ValueError:
            print("No need to reshape. For each sample, array is already 2D.")

        # Restore memory.
        (max_indices, input_memory) = self.memory
        print("Remember, the original input had shape: " + str(input_memory))
        (_, _, input_height, input_width) = input_memory

        # To prepare the input and match the memory of max indices, we use NumPy's repeat function.
        input_errors = input_errors.repeat(2, axis = 2).repeat(2, axis = 3)

        # Figure out where to send errors.
        output_errors = np.multiply(max_indices, input_errors)

        # Cut off padding.
        shaved_output_errors = output_errors[0:, 0:, 0 + self.padding : input_height - self.padding - 1,
                                                     0 + self.padding: input_width - self.padding - 1]

        # Print output errors shape.
        output_errors = np.transpose(shaved_output_errors)
        print("Output errors shape: " + str(output_errors.shape))

        # Return output errors.
        return(output_errors)