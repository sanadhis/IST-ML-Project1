import numpy as np
from scripts.helpers import batch_iter, sigmoid
from scripts.costs import compute_mse, compute_rmse, compute_loss_logistic, compute_loss_logistic_regularized
from scripts.gradients import compute_gradient_mse, compute_gradient_logistic, compute_gradient_logistic_regularized
from scripts.cross_validation import build_k_indices, cross_validation

# 1. Linear regression using gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Function to perfom gradient descent algorithm.
    Args:
        y         (numpy array): Matrix output of size N x 1.
        tx        (numpy array): Matrix input of size N x D.
        initial_w (numpy array): Initial matrix weight (parameters of the model) of size D x 1.
        max_iters (int)        : Maximum iteration for gradient descent training.
        gamma     (float)      : Step-size/learning rate constant.
    Returns:
        w    (numpy array)  : Final weight matrix by gradient descent.
        loss (float, scalar): Final cost value by gradient descent.
    """
    k_fold                           = 10
    k                                = np.random.randint(k_fold)
    seed                             = 1
    k_indices                        = build_k_indices(y, k_fold, seed)
    x_train, y_train, x_test, y_test = cross_validation(y, tx, k_indices, k)
    
    w = initial_w
    
    for n_iter in range(max_iters):
        loss_train = compute_mse(y_train, x_train, w)
        grad       = compute_gradient_mse(y_train, x_train, w)
        w          = w - gamma * grad
        loss_test  = compute_mse(y_test, x_test, w)
    return w, loss_test


# 2. Linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Function to perfom stochastic gradient descent algorithm.
    Args:
        y         (numpy array): Matrix output of size N x 1.
        tx        (numpy array): Matrix input of size N x D.
        initial_w (numpy array): Initial matrix weight (parameters of the model) of size D x 1.
        max_iters (int)        : Maximum iteration for gradient descent training.
        gamma     (float)      : Step-size/learning rate constant.
    Returns:
        w    (numpy array)  : Final weight matrix by stochastic gradient descent.
        loss (float, scalar): Final cost value by stochastic gradient descent.
    """
    k_fold                           = 10
    k                                = np.random.randint(k_fold)
    seed                             = 1
    k_indices                        = build_k_indices(y, k_fold, seed)
    x_train, y_train, x_test, y_test = cross_validation(y, tx, k_indices, k)
    
    w = initial_w
    
    for n_iter in range(max_iters):    
        for minibatch_y, minibatch_x in batch_iter(y_train, x_train, 1):
            
            loss_train = compute_mse(minibatch_y, minibatch_x, w)
            grad       = compute_gradient_mse(minibatch_y, minibatch_x, w) 
            w          = w - gamma * grad
        
        loss_test =  compute_mse(y_test, x_test, w)
    return w, loss_test


# 3. Least squares regression using normal equations
def least_squares(y, tx):
    """Function to calculate the least squares solution using normal equations.
    Args:
        y         (numpy array): Matrix output of size N x 1.
        tx        (numpy array): Matrix input of size N x D.
    Returns:
        w    (numpy array)  : Weight matrix by least squares normal equations.
        loss (float, scalar): Cost value by least squares normal equations.
    """
    k_fold                           = 10
    k                                = np.random.randint(k_fold)
    seed                             = 1
    k_indices                        = build_k_indices(y, k_fold, seed)
    x_train, y_train, x_test, y_test = cross_validation(y, tx, k_indices, k)
    
    xt        = x_train.transpose()
    w         = np.linalg.solve(xt.dot(x_train), xt.dot(y_train))
    loss_test = compute_rmse(y_test, x_test, w)
    
    return w, loss_test


# 4. Ridge regression using normal equations
def ridge_regression(y, tx, lambda_):
    """Function to calculate the least squares solution using normal equations.
    Args:
        y         (numpy array): Matrix output of size N x 1.
        tx        (numpy array): Matrix input of size N x D.
        lambda_   (float)      : Lifting constant for ridge regression.
    Returns:
        w    (numpy array)  : Weight matrix by least squares normal equations.
        loss (float, scalar): Cost value by least squares normal equations.
    """
    k_fold                           = 10
    k                                = np.random.randint(k_fold)
    seed                             = 1
    k_indices                        = build_k_indices(y, k_fold, seed)
    x_train, y_train, x_test, y_test = cross_validation(y, tx, k_indices, k)
    
    xt        = x_train.transpose()
    w         = np.linalg.solve(xt.dot(x_train) + lambda_ * (2 * y_train.shape[0]) * np.identity(xt.shape[0]), xt.dot(y_train))
    loss_test = compute_rmse(y_test, x_test, w)
    
    return w, loss_test


# 5. Logistic regression using gradient descent
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Function to perform logistic regression using gradient descent.
    Args:
        y         (numpy array): Matrix output of size N x 1.
        tx        (numpy array): Matrix input of size N x D.
        initial_w (numpy array): Initial matrix weight (parameters of the model) of size D x 1.
        max_iters (int)        : Maximum iteration for gradient descent training.
        gamma     (float)      : Step-size/learning rate constant.
    Returns:
        w    (numpy array)  : Final weight matrix by logistic regression using gradient descent.
        loss (float, scalar): Final cost value by logistic regression using gradient descent.
    """
    k_fold                           = 10
    k                                = np.random.randint(k_fold)
    seed                             = 1
    k_indices                        = build_k_indices(y, k_fold, seed)
    x_train, y_train, x_test, y_test = cross_validation(y, tx, k_indices, k)
    
    w = initial_w
    
    for n_iter in range(max_iters):
        loss      = compute_loss_logistic(y_train, x_train, w)
        grad      = compute_gradient_logistic(y_train, x_train, w)
        w         = w - gamma * grad
        loss_test = compute_loss_logistic(y_test, x_test, w)
    return w, loss_test


# 6. Regularized logistic regression using gradient descent
def reg_logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_):
    """Function to perform regularized logistic regression using gradient descent.
    Args:
        y         (numpy array): Matrix output of size N x 1.
        tx        (numpy array): Matrix input of size N x D.
        initial_w (numpy array): Initial matrix weight (parameters of the model) of size D x 1.
        max_iters (int)        : Maximum iteration for gradient descent training.
        gamma     (float)      : Step-size/learning rate constant.
        lambda    (float)      : Penalty constant.
    Returns:
        w    (numpy array)  : Final weight matrix by regularized logistic regression using gradient descent.
        loss (float, scalar): Final cost value by regularized logistic regression using gradient descent.
    """
    k_fold                           = 10
    k                                = np.random.randint(k_fold)
    seed                             = 1
    k_indices                        = build_k_indices(y, k_fold, seed)
    x_train, y_train, x_test, y_test = cross_validation(y, tx, k_indices, k)
         
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss_logistic_regularized(y_train, x_train, w, lambda_)
        grad = compute_gradient_logistic_regularized(y_train, x_train, w, lambda_)
        w    = w - gamma * grad
        loss_test = compute_loss_logistic_regularized(y_test, x_test, w, lambda_)
        
    return w, loss_test