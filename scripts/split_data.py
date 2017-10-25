import numpy as np

def split_data(x, y, ratio, myseed=1):
    """Function to split training data and test data based on given splitting ratio.
    Args:
        x      (numpy array): Original matrix input (features) of size N x D.
        y      (numpy array): Original matrix output of size N x 1.
        ratio  (float)      : Ratio of data splitting for training data with value of {0,1} with the rest (1-ratio) goes to test.
        myseed (int)        : Integer value to seed the random generator.
    Returns:
        x_train (numpy array) : Matrix input (features) for training set with size of (ratio*N) x D.
        y_train (numpy array) : Matrix output for training set with size of (ratio*N) x D.
        x_test  (numpy array) : Matrix input (features) for testing set with size of ((1-ratio)*N) x D.
        y_test  (numpy array) : Matrix output for testing set with size of ((1-ratio)*N) x D.
    """
   
    # set seed
    np.random.seed(myseed)
    
    # generate random indices
    num_row          = len(y)
    indices          = np.random.permutation(num_row)
    index_split      = int(np.floor(ratio * num_row))
    training_indices = indices[:index_split]
    test_indices     = indices[index_split:]
    
    # splitting the data
    x_train = x[training_indices]
    y_train = y[training_indices]
    x_test  = x[test_indices]
    y_test  = y[test_indices]
    
    return x_train, y_train, x_test, y_test
