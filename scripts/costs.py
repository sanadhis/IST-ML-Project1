import numpy as np
from scripts.helpers import sigmoid

def compute_mse(y, tx, w):
    """Function to compute mean square error (MSE).
    Args:
        y  (numpy array): Matrix output of size N x 1.
        tx (numpy array): Matrix input of size N x D.
        w  (numpy array): Matrix weight (parameters of the model) of size D x 1.
    Returns:
        mse (float, scalar) : The mean square error (MSE) for given model w.
    """
    
    value_error = y - tx.dot(w)
    mse         = np.inner(value_error, value_error)/(2 * y.shape[0]) 
    return mse


def compute_rmse(y, tx, w):
    """Function to compute root mean square error (RMSE).
    Args:
        y  (numpy array): Matrix output of size N x 1.
        tx (numpy array): Matrix input of size N x D.
        w  (numpy array): Matrix weight (parameters of the model) of size D x 1.
    Returns:
        rmse (float, scalar) : The root mean square error (RMSE) for given model w.
    """
    
    rmse = np.sqrt(2*compute_mse(y, tx, w))
    return rmse


def compute_loss_logistic(y, tx, w):
    """Function to compute cost of logistic regression by negative log likelihood.
    Args:
        y  (numpy array): Matrix output of size N x 1.
        tx (numpy array): Matrix input of size N x D.
        w  (numpy array): Matrix weight (parameters of the model) of size D x 1.
    Returns:
        loss (float, scalar) : The loss/cost value of logistic regression for given model w.
    """

    y_hat = sigmoid(tx.dot(w))
    loss  = (np.transpose(y)).dot(np.log(y_hat + 1e-5)) + (np.transpose(1 - y)).dot(np.log(1 - (y_hat - 1e-5)))
    loss  = np.squeeze(- loss)
    return loss


def compute_loss_logistic_regularized(y, tx, w, lambda_):
    """Function to compute cost of regularized logistic regression by negative log likelihood.
    Args:
        y       (numpy array): Matrix output of size N x 1.
        tx      (numpy array): Matrix input of size N x D.
        w       (numpy array): Matrix weight (parameters of the model) of size D x 1.
        lambda_ (float)      : Penalty constant.
    Returns:
        loss (float, scalar) : The loss/cost value of regularized logistic regression for given model w.
    """
    
    y_hat            = sigmoid(tx.dot(w))
    loss             = (np.transpose(y)).dot(np.log(y_hat + 1e-5)) + (np.transpose(1 - y)).dot(np.log(1 - (y_hat - 1e-5)))
    regularized_loss = loss - (lambda_ * np.inner(w, w))
    final_loss       = np.squeeze(- regularized_loss)
    return final_loss
