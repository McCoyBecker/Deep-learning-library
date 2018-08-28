import numpy as np
from __classes import model
from __classes.utility_funcs import d_softmax, d_cross_entropy, d_rectifier
from __classes.regularizers import L2_reg

# The AdaGrad optimizer takes in the model and the data and updates the weights for each layer.
# The weights are updated using gradient descent with a learning rate which is time - dependent.
# Gradients are kept in memory and used to update the learning rate at each training step.

class AdaGrad:

    def __init__(self, model, loss_function, initial_learning_rate, regularizer):
        self.model = model
        self.loss_function = loss_function
        self.reg_deriv = regularizer
        self.loss_deriv = eval("d_" + self.loss_function.__name__)
        self.learning_rate = initial_learning_rate
        self.gradient_memory = {}

    # Initialize gradient memory.
    def init_gradient_memory(self):
        print('\nUsing AdaGrad. Initializing gradient memory...')
        for i in range(len(self.model.layers)):
            try:
                weight_grad_memory_dimensions = self.model.layers[i].weights.shape
                bias_grad_memory_dimensions = self.model.layers[i].bias.shape
                self.gradient_memory['layer_' + str(i) + '_weight_grad'] = np.zeros(weight_grad_memory_dimensions) + 1e-3
                self.gradient_memory['layer_' + str(i) + '_bias_grad'] = np.zeros(bias_grad_memory_dimensions) + 1e-3
            except AttributeError:
                print('This is a max pooling layer. No weight gradients for this layer.')
        print('End of initialization.')

    # updateBatch takes in a new batch of data with labels.
    def updateBatch(self, batch, labels, batch_size):
        self.batch_size = batch_size
        self.batch = batch
        self.labels = labels

    # The optimize function is the workhorse of the optimizer.
    # It feeds a batch through the model and calculates the averages errors.
    # The errors are then backpropagated through the network.
    # The weights are then updated using a learning rate which is dependent on past gradient values.
    def optimize(self, initial_learning_rate):

        # Retrieve size of batch.
        num_samples = self.batch_size

        # Begin: Forward propagation.
        print("\nBeginning forward propagation...")

        # Forward propagate a batch.
        prediction = self.model.feedforward(self.batch)
        
        # Break: end of forward propagation.
        print("\n------ Forward propagation complete ------\n")

        # Initialize an array of target distributions.
        target = np.zeros((num_samples, 10))
        for i in range(num_samples):
            target[i, int(self.labels[i])] = 1
        
        # Get batch accuracy. 
        batch_accuracy = sum([1 for i in range(self.batch_size) if np.argmax(prediction[i])  == np.argmax(target[i])])/self.batch_size

        # Print batch accuracy.
        print("Batch accuracy is: " + str(batch_accuracy))

        # Calculate the average value of the loss function.
        loss_function = np.nan_to_num(np.average(self.loss_function(prediction, target)))

        # Calculate the average error over the batch.
        error = np.nan_to_num(self.loss_deriv(prediction, target)/num_samples) + self.reg_deriv(self.model, 0.5)

        # Begin: Backpropagation.
        print("\nBeginning backpropagation...")

        # Backpropagate the error.
        self.backpropagate(error, initial_learning_rate)

        # Return the accuracy and average value of the loss function.
        return((loss_function, batch_accuracy))

    # Backpropagate sends the error back through the network using the multivariate chain rule, then updates the weights.
    def backpropagate(self, error, initial_learning_rate):

        # Iterate over layers in model, backpropagating the error and updating the weights.
        for i in range(len(self.model.layers)):
            layer_index = len(self.model.layers)-1-i
            currentLayer = self.model.layers[layer_index]
            print("\nBackpropagating through layer (" +str(layer_index) +") - this layer is a " + currentLayer.__class__.__name__ + " layer.")

            # Check if current layer is a max pooling layer. If not, proceed as normal.
            if currentLayer.__class__.__name__ != 'max_pool_2D':
                (error, weight_gradients, bias_gradients) = currentLayer.backpropagate(error)

                # Print the max of the weight gradients, for diagnostic purposes.
                print("The max of the weight gradients is: " + str(np.amax(weight_gradients)))

                # Here is the special AdaGrad step: generate new learning rates for the weights and biases.
                weight_learning_rate = initial_learning_rate * 1/np.sqrt(self.gradient_memory['layer_' + str(layer_index) + '_weight_grad'])
                bias_learning_rate = initial_learning_rate * 1/np.sqrt(self.gradient_memory['layer_' + str(layer_index) + '_bias_grad'])

                # Update the weights.
                currentLayer.updateWeights(weight_gradients, bias_gradients, weight_learning_rate, bias_learning_rate)

                # Store the weight_gradient and bias_gradient in gradient memory.
                self.gradient_memory['layer_' + str(layer_index) + '_weight_grad'] += np.square(np.average(weight_gradients, axis = 0))
                self.gradient_memory['layer_' + str(layer_index) + '_bias_grad'] += np.square(np.average(bias_gradients, axis = 0))

            # If layer is max pooling, just propagate the errors through using the memory of max indices.
            else:
                error = currentLayer.backpropagate(error)