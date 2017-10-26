## Scripts
All our scripts for doing features generation and machine learning processing can be found here.

### implementations.py:
This script contain the six mandatory algorithms:
1. Linear Regression using Gradient Descent. (least_squares_GD)
2. Linear Regression using Stochastic Gradient Descent. (least_squares_SGD)
3. Linear Regression using Normal Equations. (least_squares) 
4. Ridge Regression using Normal Equations. (ridge_regression)
5. Logistic Regression using Gradient Descent. (logistic_regression)
6. Regularized Logistic Regression using Gradient Descent. (reg_logistic_regression)

### costs.py:
This script contain four functions to compute four kind of cost:
1. Mean Square Error (MSE). (compute_mse)
2. Root Mean Square Error (RMSE). (compute_rmse)
3. Loss of Logistic Regression. (compute_loss_logistic) 
4. Loss of Regularized Logistic Regression. (compute_loss_logistic_regularized)

### gradients.py:
This script contain three functions to compute three kind of gradient:
1. Gradient of Mean Square Error. (compute_gradient_mse)
2. Gradient of Logistic Regression. (compute_gradient_logistic)
3. Gradient of Regularized Logistic Regression. (compute_gradient_logistic_regularized) 

### helpers.py:
This script contain the two helper functions:
1. Sigmoid for Logistic Regression. (sigmoid)
2. Minibatch Iterator for Stochastic Gradient Descent. (batch_iter)

### preprocess.py:
This script contain our functions for data processing and features generation:
1. Polynomial basis with features generation. (standardize_with_power_terms)

### proj1_helpers.py:
This script contain the three helpers function related of reading data and making submission file:
1. Loading csv data and performing initial processing. (load_csv_data)
2. Making prediction for given weights. (predict_labels)
3. Creating submission to kaggle in csv format. (create_csv_submission) 

### cross_validation.py:
This script contain the two functions related to k-fold cross validation:
1. Building k-indices for k-fold. (build_k_indices)
2. Splitting data as training and test set for cross validation. (cross_validation)
