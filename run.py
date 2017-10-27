import numpy as np
from scripts.proj1_helpers import *
from scripts.preprocess import standardize_with_power_terms
from implementations import logistic_regression

# load data
raw_y, raw_x, ind = load_csv_data('higgs-data/train.csv')

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

def make_submission_file(x, y, w, features_reductions, filename="prediction.csv"):
    y_pred = np.ones(len(y))
    
    sets_x, sets_y, indices = create_subsets(raw_x, raw_y)

    sets_x = remove_features(sets_x, features_reductions)    

    for x, w, indices features zip(sets_x, w, indices):
        stand_x = standardize_with_power_terms(x, 2, True, with_sqrt=True)
        y_pred[indices] = predict_labels(w, stand_x)
    
    create_csv_submission(ind, y_pred, filename)

if __name__ == "__main__":
    # Define the parameters of the algorithm.
    max_iters = 5000
    gamma     = 0.000002
    lambda_   = 0.000001

    sets_x, sets_y, indices = create_subsets(raw_x, raw_y)
    
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

    sets_x = remove_features(sets_x, features_reductions)

    ws = []
    for x, y in zip(standardize_x, sets_y):
        #map y to value of either zero or one
        mapped_y = (y+1)/2
        
        initial_w = np.zeros(x.shape[1])
        w, loss   = logistic_regression(mapped_y, x, initial_w, max_iters, gamma)
        
        ws.append(w)
    
    test_y, test_x, ind = load_csv_data('higgs-data/test.csv')

    make_submission_file(test_x, test_y, ws, features_reductions, "final-submission.csv")
    