import numpy as np

# Define the derivative of L2 parameter regularization.

def L2_reg (model, lagrange):
    output = 0
    for layer in model.layers:
        if layer.__class__.__name__ != 'max_pool_2D':
            output += np.sum(2*np.abs(layer.weights))
    return(lagrange*output)