import numpy as np
from scripts.proj1_helpers import *
from scripts.preprocess import generate_features
from implementations import logistic_regression

def print_banner(message):
    """Function to print message in nicer way.
    Args:
        message (string): The message
    Returns:
        void
    """

    print("#############################################################################")
    print(message)
    
def create_subsets(x, y):
    """Function to divide dataset into 8 subsets based on PRI_JET_NUM and DER_MASS_MMC.
    Args:
        x       (numpy array)         : Matrix features of size N x D.
        y       (numpy array)         : Matrix output of size N x 1.        
    Returns:
        sets_x  (list of numpy arrays): List contains 8 subsets data from matrix features x.
        sets_y  (list of numpy arrays): List contains 8 subsets data from matrix output y.
        indices (list of numpy arrays): List contains 8 subsets of indices for each subset.
    """
    # initiate empty list for return variables.
    sets_x = []
    sets_y = []
    indices = []

    # iterate through value of PRI_JET_NUM (ranged inclusively from 0 until 3)
    for pri_jet_num_val in np.unique(x[:,22]):
        
        # Find subset which DER_MASS_MMC is not equal to -999
        mask = (x[:,22] == pri_jet_num_val) & (x[:,0] != -999)
        x_tmp   = x[mask,:]
        y_tmp   = y[mask]

        # store the subset into list
        sets_x.append(x_tmp)
        sets_y.append(y_tmp)
        indices.append(mask)

        # Find subset which DER_MASS_MMC is equal to -999
        mask = (x[:,22] == pri_jet_num_val) & (x[:,0] == -999)
        x_tmp   = x[mask,:]
        y_tmp   = y[mask]

        # store the subset into list
        sets_x.append(x_tmp)
        sets_y.append(y_tmp)
        indices.append(mask)        
        
    # return subsets of x, y, and corresponding indices
    return sets_x, sets_y, indices

def remove_features(sets_x, unused_features):
    """Function to remove insignificant features for each subset of x.
    Args:
        sets_x           (list of numpy arrays): List contains 8 subsets data from matrix features x.    
        unused_features  (list of int)         : List contains unused features for each subset of x.
    Returns:
        significant_x    (list of numpy arrays): List contains 8 subsets data of x with significant features only.
    """

    # initiate empty list for return variable
    significant_x = []    

    # iterate through subsets and their corresponding insignificant features
    for x, features in zip(sets_x, unused_features):
        # remove features from subset and store the result into list
        significant_x.append(np.delete(x,features,1))
        
    return significant_x

def standardize(sets_x):
    """Function to generate final features matrix for ML methods.
    Features matrix will be normalized using standard score (z-score) and 
    will be expanded using logarithmic and polynomial basis function.
    Args:
        sets_x         (list of numpy arrays): List contains 8 subsets data from matrix features x.        
    Returns:
        standardized_x (list of numpy arrays): List contains 8 normalized subsets with features data expanded from matrix features x.            
    """

    # initiate empty list for return variable
    standardized_x = []

    # iterate through subsets
    for x in sets_x:
        # call preprocess function, normalize and generate features for each subset
        # and store the result into list
        standardized_x.append(generate_features(x, 2, True, with_log=True))

    return standardized_x

def make_submission_file(w, unused_features, filename="prediction.csv"):
    """Function to produce final csv submission file to Kaggle.
    Args:
        w                (list of numpy array)              : List contains 8 models for classifying each subset of x.
        unused_features  (list of int)                      : List contains unused features for each subset of x.
        filename         (string, default "prediction.csv") : The name of final submission file.  
    Returns:
        void
    """

    # load test datasets
    print_banner("7. Read test dataset from higgs-data/test.csv")                    
    test_y, test_x, ind = load_csv_data('higgs-data/test.csv')

    # Construct Matrix Output with values of one
    y_pred = np.ones(len(test_y))

    # Split test dataset based
    print_banner("8. Split the test dataset into 8 subsets")  
    test_sets_x, _, indices = create_subsets(test_x, test_y)

    # Remove features of test datasets based on PRI_JET_NUM and DER_MASS_MMC
    print_banner("9. Remove features in each test subset based on PRI_JET_NUM and DER_MASS_MMC")
    test_sets_x = remove_features(test_sets_x, unused_features)    

    # Iterate through the test subsets with their models accordingly
    print_banner("10. Predict each test subset using their corresponding model")              
    for x, w, index in zip(test_sets_x, w, indices):

        # Perform z-score standardization and transform matrix features of test data into polynomial basis   
        stand_x = generate_features(x, 2, True, with_log=True)

        # Get the prediction
        y_pred[index] = predict_labels(w, stand_x)

        print_banner("  Predicting subset: DONE")        
    
    # Creating submission file
    print_banner("11. Making final submission file with csv format")     
    create_csv_submission(ind, y_pred, filename)

# ---------- Main function goes here ----------
if __name__ == "__main__":
    # Define the static values of the algorithm.
    max_iters = 5000
    gamma     = 0.000002
    lambda_   = 0.000001

    # load train datasets
    print_banner("1. Read train data from higgs-data/train.csv")
    raw_y, raw_x, ind = load_csv_data('higgs-data/train.csv')

    """
        Based on PRI_JET_NUM (feature 22), which ranged in value of inclusive [0,3], we devide the training 
        data into 4 groups. From these 4 groups, we devide again each of them into 2 subsets based on outliers (-999) 
        value in DER_MASS_MMC (feature 1). So these approach give us 8 subsets to train and to obtain the 8 
        corresponding models (weights).
    """
    print_banner("2. Split train dataset into 8 subsets")          
    sets_x, sets_y, indices = create_subsets(raw_x, raw_y)

    """    
        After analyzing correlation between features in datasets, 
        we decide to reduces features based on the values of PRI_JET_NUM and DER_MASS_MMC as follows:
        
        if PRI_JET_NUM = 0 and DER_MASS_MMC != -999, drop features 4, 5, 6, 11, 12, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29 
        if PRI_JET_NUM = 0 and DER_MASS_MMC == -999, drop features 0, 4, 5, 6, 11, 12, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29 
        if PRI_JET_NUM = 1 and DER_MASS_MMC != -999, drop features 4, 5, 6, 11, 12, 15, 18, 20, 22, 26, 27, 28 
        if PRI_JET_NUM = 1 and DER_MASS_MMC == -999, drop features 0, 4, 5, 6, 11, 12, 15, 18, 20, 22, 26, 27, 28 
        if PRI_JET_NUM == 2 and DER_MASS_MMC != -999, drop features 11, 15, 18, 20, 22, 28
        if PRI_JET_NUM == 2 and DER_MASS_MMC == -999, drop features 0, 11, 15, 18, 20, 22, 28
        if PRI_JET_NUM == 3 and DER_MASS_MMC != -999, drop features 11, 15, 18, 20, 22, 28
        if PRI_JET_NUM == 3 and DER_MASS_MMC == -999, drop features 0, 11, 15, 18, 20, 22, 28
    """
    features_reductions = [
                            [4, 5, 6, 11, 12, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29],
                            [0, 4, 5, 6, 11, 12, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29],
                            [4, 5, 6, 11, 12, 15, 18, 20, 22, 26, 27, 28],
                            [0, 4, 5, 6, 11, 12, 15, 18, 20, 22, 26, 27, 28],
                            [11, 15, 18, 20, 22, 28],
                            [0, 11, 15, 18, 20, 22, 28],
                            [11, 15, 18, 20, 22, 28],
                            [0, 11, 15, 18, 20, 22, 28]
                          ]
    print_banner("3. Remove features in each train subset based on PRI_JET_NUM & DER_MASS_MMC")          
    sets_x = remove_features(sets_x, features_reductions)

    # Perform z-score standardization and transform matrix features into polynomial basis and logarithmic basis
    print_banner("4. Standardize, Perform logarithmic & polynomial basis function into each matrix features of train subset")              
    sets_x = standardize(sets_x)

    # Store eight weights into list ws
    ws = []

    print_banner("5. Begin classification with logistic regression, print test accuracy for each model")          
    # Iterate through each subsets
    for x, y in zip(sets_x, sets_y):
        # map y to value of either zero or one
        mapped_y = (y+1)/2
        
        # Set initial weights as zero matrix of size D x 1
        initial_w = np.zeros(x.shape[1])

        # Use logistic regression
        w, loss   = logistic_regression(mapped_y, x, initial_w, max_iters, gamma)
        
        # Checking model accuracy for each subsets
        test_accuracy = (np.mean(predict_labels(w, x) == y))
        print_banner("  Accuracy of model for given training data subset: " + str(test_accuracy))

        # store models to ws
        ws.append(w)

    # Produce the final-submission file
    print()
    print_banner("Apply models to test data and produce final submission file")
    make_submission_file(ws, features_reductions, "final-submission.csv")
