import numpy as np

# Import utility functions and layers.

from __classes import layers
from __classes import utility_funcs
from __classes import data_processing_funcs

# A model class. Stores layers and performs propagation of input data. Feeds into the optimizer for training.
class model:

    def __init__(self):
        self.layers = []

    # Add a layer to the model.
    def addLayer(self, layer):
        self.layers.append(layer)

    # Propagate a data batch through the model.
    def feedforward(self, inputData):
        i = 0
        for layer in self.layers:
            print("\nForward propagating through layer (" + str(i) +") - this layer is a " + layer.__class__.__name__ + " layer.")
            inputData = layer.propagate(inputData)
            i+=1
        return(inputData)

    # Prints Keras-style summary of model parameters and architecture. IMPLEMENT.
    def summary(self):
        return(0)