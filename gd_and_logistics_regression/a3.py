__author__ = "Mackenzie Clark"
__date__ = "Mar 22nd 2019"

import numpy as np
from matplotlib import pyplot as plt
from data.data_utils import load_dataset
import time
import random

possible_datasets = ["pumadyn32nm"]
learning_rates = [0.0001, 0.001, 0.01, 0.1]
gd_types = ["SGD", "GD"]

################ functions copied from a2 ######################

def load_data(dataset):
    '''
    load the data from the mauna_loa set to create a GLM from
    '''
    if dataset == 'rosenbrock':
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', n_train=1000, d=2)
    else:
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(str(dataset))
    return x_train, x_valid, x_test, y_train, y_valid, y_test

def l_2_norm(x1, x2):
    '''
    compute the l-2 norm of a vector x2 with vector x1
    note: if the vectors are different lenght (which they shouldn't be),
        we take the smallest length of vector
    '''
    return np.linalg.norm([x1-x2], ord=2)

def rmse(y_estimates, y_valid):
    '''
    calculate the root mean squared error between the estimated y values and 
        the actual y values

    param y_estimates: list of ints
    param y_valid: list of ints, actual y values
    return: float, rmse value between two lists
    '''
    return np.sqrt(np.average(np.abs(y_estimates-y_valid)**2))

def make_x_matrix(x_data):
    '''
    takes a vetor of data (matrix) x_data and adds a column of 1's
    at the front
    '''
    X = np.ones((len(x_data), len(x_data[0]) + 1))
    X[:, 1:] = x_data
    return X

###################### main a3 functions ########################

def linear_reg(x_train, x_valid, x_test, y_train, y_valid, y_test):
    '''
    function computes the optimal weights of the model created with 
    linear regression based on the input data
    returns: optimal test rmse, optimal weight vector
    '''
    x_train_valid = np.vstack([x_train, x_valid])
    y_train_valid = np.vstack([y_train, y_valid])

    # limit training to first 1000 pts
    x_train_valid = x_train_valid[:1000]
    y_train_valid = y_train_valid[:1000]

    # create the x matrix to SVD with a column of 1's to start
    X = make_x_matrix(x_train_valid)

    # compute the SVD
    U, s, V = np.linalg.svd(X)
    # make large Sigma matrix - full SVD
    Sigma = np.vstack([np.diag(s), np.zeros((len(x_train_valid) - len(s), len(s)))])
    # using pseduo-inverse
    S_inv = np.linalg.pinv(Sigma)

    # calculate the weights, V.T*Sigma-1*U.T*y
    w = np.dot(V.T, np.dot(S_inv, np.dot(U.T, y_train_valid)))

    # create predictions
    x_test_new = make_x_matrix(x_test)
    y_pred = np.dot(x_test_new, w)

    test_rmse = rmse(y_pred, y_test)
    return test_rmse, w

def grad_desc(x_train, x_test, y_train, y_test, reg_rmse, reg_w, gd_type = "GD"):
    '''
    computes the full-batch gradient descent for data provided in the
    arguments of the function

    param reg_rmse: float, the test RMSE calculated from linear regression
    param reg_w: vector, the model weights calculated with linear regression
    param gd_type: str, either "GD" or "SGD" to signify full-batch or stochastic
    '''
    if gd_type == "GD":
        rates = learning_rates[1:]
    else:
        rates = learning_rates[:-1]
    # only using 1000 points of the training data
    x_train = x_train[:1000]
    y_train = y_train[:1000]

    X = make_x_matrix(x_train)
    X_test = make_x_matrix(x_test)

    final_w = []
    test_rmses = []
    losses = {}         # so that we can plot loss for all learning rates

    for r in rates:

        losses[r] = []
        # initialize w/minimizer array to zeros
        w = np.zeros(len(X[0]))

        for i in range(1000):   # arbitrary num iterations
            # make predictions with current minimizer values
            y_pred = np.dot(X, w)

            if gd_type == "GD":
                # use all the training points
                grad = np.insert(np.zeros(np.shape(x_train[0])), 0, 0)
                for t in range(len(y_pred)):
                    grad += 2* (y_pred[t] - y_train[t]) * np.insert(x_train[t], 0, 1)
                grad = grad/len(y_pred)
            else:       # stochastic
                # select a point t randomly (mini-batch 1)
                t = random.randint(0, len(y_pred) - 1)
                grad = 2 * (y_pred[t] - y_train[t]) * np.insert(x_train[t], 0, 1)
            
            # ensure shape is a column vector
            grad = grad.reshape((len(X[0]), 1))

            # update our weight estimates based on the gradient & learning rate
            w = np.add(w, -r*grad)

            # compute full-batch loss, RMSE ^2
            losses[r].append(rmse(y_pred, y_train) ** 2)
    
        
        # after the final iteration, compute the test RMSE
        y_pred_test = np.dot(X_test, w)
        test_rmses.append(rmse(y_pred_test, y_test))
        final_w.append(w)

    # now find the optimal learning rate + rmse
    min_test_rmse = min(test_rmses)
    min_w = final_w[test_rmses.index(min_test_rmse)]
    preferred_rate = rates[test_rmses.index(min_test_rmse)]
    
    return losses, min_test_rmse, min_w, preferred_rate

def plot_losses(x_train, y_train, losses, title, opt_w, figure_num):
    '''
    function that plots the gradient descent losses over time

    param losses: list, rmse validation losses from GD or SGD
    param title: str, plot title
    param opt_w: vector, optimal weights from the linear regression SVD 
    param figure_num: int, unique for all figures that are created
    '''
    # using the linear regression SVD opt_w to compare
    X = make_x_matrix(x_train[:1000])
    y_train = y_train[:1000]
    y_pred = np.dot(X, opt_w)
    optimal_loss = rmse(y_pred, y_train) ** 2
    optimal_loss_pts = [optimal_loss] * 1000
    iterations = range(1,1001)

    plt.figure(figure_num)
    plt.title(title + " Full-Batch Validation Loss vs. Iteration #")
    plt.plot(iterations, optimal_loss_pts, '-b', label='Lin. regression loss (optimal)')
    plt.xlabel('Iterations')
    plt.ylabel('Full-Batch Validation Loss')
    for rate in losses:
        plt.plot(iterations, losses[rate], label='Loss for rate ' + str(rate))
    plt.legend(loc='upper right')
    plt.savefig(title + '_loss.png')

def log_likelihood(X, y, w):
    '''
    function that computes the log loss formula
    param X: matrix
    param y, w: vectors
    '''
    log_p = np.dot(y.T, np.log(sigmoid(X))) + np.dot(np.subtract(1, y).T, np.log(np.subtract(1, sigmoid(X))))
    # get rid of the double array by typecasting as a float
    return float(log_p)

def sigmoid(x):
    '''
    calculate the sigmoid for a vector x
    '''
    sig = np.divide(1, np.add(1, np.exp(x)))
    if str(type(sig)) != "<class 'numpy.float64'>":
        sig = np.reshape(sig, (np.shape(sig)[0], 1))
    return sig

def log_grad_desc(x_train, x_valid, x_test, y_train, y_valid, y_test, gd_type="GD"):
    '''
    compute the classification with log
    '''
    rates = learning_rates[:-1]
    x_train = np.vstack([x_train, x_valid])
    y_train = np.vstack([y_train, y_valid])

    X = make_x_matrix(x_train)
    X_test = make_x_matrix(x_test)

    # keep track of multiple test methods, test accuracy and test log-likelihood
    test_accuracy = []
    test_logl = []
    negative_ll = {}

    for r in rates:
        negative_ll[r] = []
        # initialize w/minimizer array to zeros
        w = np.zeros(len(X[0]))

        for i in range(2000):
            y_pred = np.dot(X, w)

            # now estimate the losses for SGD and GD for this iteration/w
            if gd_type == "GD":
                grad = np.zeros(np.shape(w))
                for t in range(len(y_train)):
                    grad += (y_train[t] - sigmoid(y_pred[t])) * X[t, :]
            else:
                # 1x mini-batch GD
                t = random.randint(0, len(x_train) - 1)
                grad = (y_train[t] - sigmoid(y_pred[t])) * X[t, :]

            w = np.add(w, -r*grad)

            negative_ll[r].append(-1*log_likelihood(y_pred, y_train, w))

        y_test_pred = np.dot(X_test, w)
        test_logl.append(log_likelihood(y_test_pred, y_test, w))
        # compute the predictions from the sigmoid
        y_pred = sigmoid(y_test_pred)
        y_pred = np.reshape(y_pred, np.shape(y_test))

        for j in range(len(y_pred)):
            if y_pred[j] > 0.5:
                y_pred[j] = 1
            elif y_pred[j] < 0.5:
                y_pred[j] = 0
            else:
                y_pred[j] = -1

        test_accuracy.append((y_test == y_pred).sum() / len(y_test))

    #print(test_logl)
    opt_log = min(test_logl)
    opt_acc = max(test_accuracy)
    opt_log_rate = rates[test_logl.index(opt_log)]
    opt_acc_rate = rates[test_accuracy.index(opt_acc)]

    return opt_log, opt_acc, opt_log_rate, opt_acc_rate, negative_ll

def plot_logistic_losses(neg_ll, title, figure_num):
    '''
    '''
    iterations = range(1,2001)

    plt.figure(figure_num)
    plt.title(title + " Negative Log Likelihood vs. Iteration #")
    plt.xlabel('Iterations')
    plt.ylabel('Negative Log Likelihood')
    for rate in neg_ll:
        plt.plot(iterations, neg_ll[rate], label='Likelihood for rate ' + str(rate))
    plt.legend(loc='upper right')
    plt.savefig(title + '_likelihood.png')

if __name__ == "__main__":
    Q1 = False
    Q2 = True

    if Q1:
        # first perform linear regression to determine optimal weights of the model
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_data("pumadyn32nm")
        lin_reg_rmse, lin_reg_w = linear_reg(x_train, x_valid, x_test, y_train, y_valid, y_test)
        print("optimal RMSE from linear regression: " + str(lin_reg_rmse))
        
        # use the weights and RMSE computed in linear regression as benchmarks
        gd_losses, gd_rmse, gd_w, gd_rate = grad_desc(x_train, x_test, y_train, y_test, lin_reg_rmse, lin_reg_w)
        plot_losses(x_train, y_train, gd_losses, "GD", lin_reg_w, 1)
        print("########## Full-Batch GD #########")
        print("best learning rate: " + str(gd_rate))
        print("gives this rmse: " + str(gd_rmse))

        sgd_losses, sgd_rmse, sgd_w, sgd_rate = grad_desc(x_train, x_test, y_train, y_test, lin_reg_rmse, lin_reg_w, "SGD")
        plot_losses(x_train, y_train, sgd_losses, "SGD", lin_reg_w, 2)
        print("########## Mini-Batch SGD #########")
        print("best learning rate: " + str(sgd_rate))
        print("gives this rmse: " + str(sgd_rmse))

    if Q2:
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_data("iris")
        y_train, y_valid, y_test = y_train[:,(1,)], y_valid[:,(1,)], y_test[:,(1,)]

        # turn all the booleans into integers 
        y_train = np.asarray(y_train, int)
        y_valid = np.asarray(y_valid, int)
        y_test = np.asarray(y_test, int)

        # classification gradient descent with accuracy and log likelihood erro measurements
        opt_ll, opt_acc, opt_ll_rate, opt_acc_rate, negative_ll = log_grad_desc(x_train, x_valid, x_test, y_train, y_valid, y_test)
        plot_logistic_losses(negative_ll, "GD", 3)
        print("########## Full-Batch Logistic Regression #########")
        print("best accuracy: " + str(opt_acc))
        print("rate for accuracy: " + str(opt_acc_rate))
        print("best log likelihood: " + str(opt_ll))
        print("rate for log likelihood: " + str(opt_ll_rate))

        sopt_ll, sopt_acc, sopt_ll_rate, sopt_acc_rate, snegative_ll = log_grad_desc(x_train, x_valid, x_test, y_train, y_valid, y_test, "SGD")
        plot_logistic_losses(negative_ll, "SGD", 2)
        print("########## Stochastic Logistic Regression #########")
        print("best accuracy: " + str(sopt_acc))
        print("rate for accuracy: " + str(sopt_acc_rate))
        print("best log likelihood: " + str(sopt_ll))
        print("rate for log likelihood: " + str(sopt_ll_rate))