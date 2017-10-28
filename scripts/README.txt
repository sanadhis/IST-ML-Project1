Scripts
All scripts with functions and helpers to perform our machine learning project.

### costs.py
contains four functions to compute four kind of cost:
1. compute_mse : (Mean Square Error (MSE))
2. compute_rmse : (Root Mean Square Error (RMSE))
3. compute_loss_logistic : (Loss of Logistic Regression) 
4. compute_loss_logistic_regularized : (Loss of Regularized Logistic Regression)

### cross_validation.py
This script contain the two functions related to k-fold cross validation:
1. build_k_indices : (Building k-indices for k-fold)
2. cross_validation : (Splitting data as training and test set for cross validation)

### gradients.py
contains three functions to compute three kind of gradient:
1. compute_gradient_mse : (Gradient of Mean Square Error)
2. compute_gradient_logistic : (Gradient of Logistic Regression)
3. compute_gradient_logistic_regularized : (Gradient of Regularized Logistic Regression)

### helpers.py
contains the two helper functions:
1. sigmoid : (Sigmoid for Logistic Regression)
2. batch_iter : (Minibatch Iterator for Stochastic Gradient Descent)

### preprocess.py
contains our functions for features processing and features generation:
1. normalize_data : (Data normalization with standard score/z-score)
2. clean_data : (Removing outliers {-999} from dataset)
3. log_basis : (Features generation using logarithmic basis)
4. polynomial_basis : (Features generation using polynomial basis)
5. generate_features : (Wrap-up function for the first 4 functions)

### proj1_helpers.py
contains the three helpers function related of reading data and making submission file:
1. load_csv_data : (Loading csv data and performing initial processing)
2. predict_labels : (Making prediction for given weights)
3. create_csv_submission : (Creating submission to kaggle in csv format) 
