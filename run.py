import numpy as np
from scripts.proj1_helpers import *
from scripts.preprocess import standardize_with_power_terms
from implementations import logistic_regression

def print_banner(message):
    print("#############################################################################")
    print(message)
    
def create_subsets(x, y):
    sets_x = []
    sets_y = []
    indexes = []
    for pri_jet_num_val in np.unique(x[:,22]):
        
        indices = (x[:,22] == pri_jet_num_val) & (x[:,0] != -999)
        x_tmp   = x[indices,:]
        y_tmp   = y[indices]

        sets_x.append(x_tmp)
        sets_y.append(y_tmp)
        indexes.append(indices)

        indices = (x[:,22] == pri_jet_num_val) & (x[:,0] == -999)
        x_tmp   = x[indices,:]
        y_tmp   = y[indices]

        sets_x.append(x_tmp)
        sets_y.append(y_tmp)
        indexes.append(indices)        
        
    return sets_x, sets_y, indexes

def remove_features(sets_x, unused_features):
    l = []    
    for x, features in zip(sets_x, unused_features):
        l.append(np.delete(x,features,1))
    return l

def standardize(sets_x):
    l = []
    for x in sets_x:
        l.append(standardize_with_power_terms(x, 2, True, with_sqrt=True))
    return l

def make_submission_file(w, features_reductions, filename="prediction.csv"):
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
    test_sets_x = remove_features(test_sets_x, features_reductions)    

    # Iterate through the test subsets with their models accordingly
    print_banner("10. Predict each test subset using their corresponding model")              
    for x, w, index in zip(test_sets_x, w, indices):

        # Perform z-score standardization and transform matrix features of test data into polynomial basis   
        stand_x = standardize_with_power_terms(x, 2, True, with_sqrt=True)

        # Get the prediction
        y_pred[index] = predict_labels(w, stand_x)

        print_banner("  Predicting subset: DONE")        
    
    # Creating submission file
    print_banner("11. Making final submission file with csv format")     
    create_csv_submission(ind, y_pred, filename)

if __name__ == "__main__":
    # Define the static values of the algorithm.
    max_iters = 5000
    gamma     = 0.000002
    lambda_   = 0.000001

    # load train datasets
    print_banner("1. Read train data from higgs-data/train.csv")
    raw_y, raw_x, ind = load_csv_data('higgs-data/train.csv')

    """
        Based on PRI_JET_NUM (feature 22), which ranged in value of inclusive [0,4], we devide the training 
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

    # Perform z-score standardization and transform matrix features into polynomial basis
    print_banner("4. Standardize and Perform polynomial basis into each matrix features of train subset")              
    sets_x = standardize(sets_x)

    # Store eight weights into list ws
    ws = []

    print_banner("5. Begin training with logistic regression, print test accuracy for each model")          
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
