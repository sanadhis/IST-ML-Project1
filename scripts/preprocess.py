import numpy as np

def normalize_data(x):
    """Function to normalize data using standard score / z-score.
    Args:
        x            (numpy array): Matrix features of size N x D.
    Returns:
        normalized_x (numpy array): Matrix features of size N x D, with its data normalized.
    """
    
    # Normalization using Standard Score
    # Standardize data by substracting with mean and dividing with standard deviation
    mean         = np.mean(x, axis = 0)
    std_dev      = np.std(x, axis = 0)
    normalized_x = (x - mean) / std_dev

    return normalized_x

def clean_data(x):
    """Function to clean outliers (-999 value) in matrix features and replace them with median.
    Args:
        x (numpy array): Matrix features of size N x D.
    Returns:
        x (numpy array): Matrix features of size N x D, with outliers replaced by median of each feature.
    """
    
    # locate useful values and outliers (-999)
    mask     = (x != -999)
    outliers = (~mask)

    # Replacing outliers (-999) with median (per feature) of the rest of the data
    x[outliers] = (x[outliers] * 0) + np.median(x[mask], axis = 0)
    
    return x

def log_basis(x):
    """Function to generate logarithmic basis function for matrix features of x.
    Args:
        x              (numpy array): Matrix features of size N x D.
    Returns:
        log_features_x (numpy array): Matrix log features of size N x 2D.
    """
    
    # Find minimum value per feature
    min_               = np.min(x, axis=0)
    
    # Generate log feature of ( 1 / (1 + log(1+x-min)) )
    log_x_1            = 1 / (1 + np.log(1 + x - min_))
    
    # normalized log feature 1
    normalized_log_x_1 = normalize_data(log_x_1)
    
    # Generate log feature of ( log(1+x-min) )
    log_x_2            = np.log(1 + x - min_)
    
    # normalized log feature 2
    normalized_log_x_2 = normalize_data(log_x_2)
    
    # combine log feature 1 and log feature 2 as one single numpy array
    log_features_x = np.concatenate((normalized_log_x_1, normalized_log_x_2), 1)
    
    return log_features_x

def polynomial_basis(x, degree):
    """Function to generate polynomial basis function for matrix features of x.
    Args:
        x               (numpy array)        : Matrix features of size N x D.
        degree          (int, scalar)        : Degree of polynomial basis function. 
    Returns:
        poly_features_x (list of numpy array): Matrices polynomial features with each size of N x ((degree-1) * D).
    """
    
    # locate useful values and outliers (-999)
    mask     = (x != -999)
    outliers = (~mask)
    
    # for power>= 2, store each polynomial basis features into list
    poly_features_x = []

    # iterate through each degree, 2 <= degree < power+1
    for power in range(2, degree + 1):
        # Generate matrix power of features
        poly_x           = np.power(x, power)

        # Find mean of non-outlier values
        mean_poly_x      = np.sum( poly_x * mask, axis=0)/ np.sum(mask, axis=0)
        
        # Find standard deviation of non-outlier values
        std_dev_poly_x   = np.sqrt( np.sum( ((poly_x - mean_poly_x) * mask)**2, axis=0)/np.sum(mask, axis=0))
        
        # Normalization using Standard Score, apply to non-outlier values only
        # Standardize data by substracting with mean and dividing with standard deviation
        poly_x           = (poly_x * mask - mean_poly_x)/std_dev_poly_x
        
        # Replace outliers by zero
        poly_x[outliers] = 0
        
        # add to list
        poly_features_x.append(poly_x)

    return poly_features_x

def generate_features(x, degree, with_ones = True, with_log = False):
    """Function to standardize data input and generate features.
    Args:
        x              (numpy array)            : Matrix features of size N x D.
        degree         (int, scalar)            : Degree of polynomial basis function.
        with_ones      (boolean, default true)  : Boolean value to indicate either adding feature column with ones value or not.
        with_logs      (boolean, default false) : Boolean value to indicate either generating features with log basis function or not.
    Returns:
        preprocessed_x (numpy array)            : Final matrix features to be processed using ML methods. Minimal size of N x D.
    """
    
    # Remove outliers from matrix features
    x_without_outliers = clean_data(x)

    # normalize matrix features (without outliers) using standard score (z-score)
    normalized_x       = normalize_data(x_without_outliers)
    
    # Form matrix features at least with normalized features
    preprocessed_x     = normalized_x

    # if generating with logarithmic basis function
    if with_log:
        # generate logarithmic basis function for matrix input x
        log_features_x = log_basis(x)

        # combine logarithmic basis features into final matrix
        preprocessed_x = np.concatenate((preprocessed_x, log_features_x), axis = 1)

    # generate polynomial basis function for matrix input x
    poly_features_x    = polynomial_basis(x, power)
    
    # iterate through the polynomial features of x (through degree)
    for poly_x in poly_features_x:
        preprocessed_x = np.concatenate((preprocessed_x, poly_x), axis=1)
    
    # if adding feature column with ones value.
    if with_ones:
        # ---------- Add ones to the matrix --------
        tmp       = np.ones([preprocessed_x.shape[0], preprocessed_x.shape[1] + 1])
        tmp[:,1:] = preprocessed_x
        preprocessed_x   = tmp

    return preprocessed_x