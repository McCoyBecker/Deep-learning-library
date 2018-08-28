import numpy as np

# Define the batch sample function which prepares batches for training.
def batch_sample(inputData, labels, batch_size):

    # Retrieve shape of input data.
    (num_samples, sample_height, sample_width) = inputData.shape

    # Input image depth.
    input_depth = 1
    
    # Initialize output batch and labels.
    output_train = np.zeros((batch_size, input_depth, sample_height, sample_width))
    output_labels = np.zeros(batch_size)

    # Generate random sample of indices without replacement.
    sample_list = np.random.choice([i for i in range(num_samples)], batch_size, replace = False)

    # Generate output batch.
    for i in range(batch_size):
        output_train[i, input_depth-1] = inputData[sample_list[i]]
        output_labels[i] = labels[sample_list[i]]

    # Return output.
    return(output_train, output_labels, batch_size)

# A utility function for padding input into max pooling.
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector