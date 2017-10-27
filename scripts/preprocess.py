import numpy as np

def normalize_data(x):
    # Normalization using Standard Score
    # Standardize data by substracting with mean and dividing with standard deviation
    mean         = np.mean(x, axis = 0)
    std_dev      = np.std(x, axis = 0)
    normalized_x = (x - mean) / std_dev

    return normalized_x

def clean_data(x):
    # locate useful values and outliers (-999)
    mask     = (x != -999)
    outliers = (~mask)

    # Replacing outliers (-999) with median of the rest of the data
    x[outliers] = (x[outliers] * 0) + np.median(x[mask], axis = 0)
    return x

def log_basis(x):
    min_               = np.min(x, axis=0)
    log_x_1            = 1 / (1 + np.log(1 + x - min_))
    normalized_log_x_1 = normalize_data(log_x_1)
    log_x_2            = np.log(1 + x - min_)
    normalized_log_x_2 = normalize_data(log_x_2)
    
    log_features_x = np.concatenate((normalized_log_x_1, normalized_log_x_2), 1)
    return log_features_x

def polynomial_basis(x, power):
    # locate useful values and outliers (-999)
    mask     = (x != -999)
    outliers = (~mask)

    poly_features_x = []

    for degree in range(2, power + 1):
        poly_x           = np.power(x, degree)

        mean_poly_x      = np.sum( poly_x * mask, axis=0)/ np.sum(mask, axis=0)
        std_dev_poly_x   = np.sqrt( np.sum( ((poly_x - mean_poly_x) * mask)**2, axis=0)/np.sum(mask, axis=0))

        poly_x           = (poly_x * mask - mean_poly_x)/std_dev_poly_x
        poly_x[outliers] = 0
        poly_features_x.append(poly_x)

    return poly_features_x

def generate_features(x, power, with_ones = True, impute_with = 'mean', with_log = False):
    
    x_without_outliers = clean_data(x)
    normalized_x       = normalize_data(x_without_outliers)
    
    preprocessed_x     = normalized_x

    if with_log:
        log_features_x = log_basis(x)
        preprocessed_x = np.concatenate((preprocessed_x, log_features_x), axis = 1)

    poly_features_x    = polynomial_basis(x, power)
    
    for poly_x in poly_features_x:
        preprocessed_x = np.concatenate((preprocessed_x, poly_x), axis=1)
    
    # ---------- Add ones to the matrix --------
    if with_ones:
        tmp       = np.ones([preprocessed_x.shape[0], preprocessed_x.shape[1] + 1])
        tmp[:,1:] = preprocessed_x
        preprocessed_x   = tmp

    return preprocessed_x