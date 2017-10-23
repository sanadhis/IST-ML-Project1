import numpy as np
from scripts.helpers import sigmoid

def compute_gradient_mse(y, tx, w):
    """Function to compute gradient of mse for given model w.
    Args:
        y  (numpy array): Matrix output of size N x 1.
        tx (numpy array): Matrix input of size N x D.
        w  (numpy array): Matrix weight (parameters of the model) of size D x 1.
    Returns:
        gradient (numpy array) : Matrix Gradient of size D x 1.
    """
    
    value_error = y - tx.dot(w)
    gradient    = -(np.transpose(tx).dot(value_error)) / (y.shape[0])
    return gradient


def compute_gradient_logistic(y, tx, w):
    """Function to compute gradient of loss of logistic regression for given w.
    Args:
        y  (numpy array): Matrix output of size N x 1.
        tx (numpy array): Matrix input of size N x D.
        w  (numpy array): Matrix weight (parameters of the model) of size D x 1.
    Returns:
        gradient (numpy array) : Matrix Gradient of size D x 1.
    """
    
    y_hat = sigmoid(tx.dot(w))
    gradient = (np.transpose(tx)).dot(y_hat - y)
    return gradient


def compute_gradient_logistic_regularized(y, tx, w, lambda_):
    """Function to compute gradient of loss of regularized logistic regression for given w.
    Args:
        y       (numpy array): Matrix output of size N x 1.
        tx      (numpy array): Matrix input of size N x D.
        w       (numpy array): Matrix weight (parameters of the model) of size D x 1.
        lambda_ (float)      : Penalty constant.
    Returns:
        gradient (numpy array) : Matrix Gradient of size D x 1.
    """
    
    y_hat                = sigmoid(tx.dot(w))
    gradient             = (np.transpose(tx)).dot(y_hat - y)
    regularized_gradient = gradient + (2 * lambda_ * w)
    return regularized_gradient